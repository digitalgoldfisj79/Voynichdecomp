# RUNNING RESULTS — pharma drawings ↔ labels, same-plant pairs, T4/T3/T2b (session 2026-11-26, started 06:31 local)
TIME NOTE: entries are labelled by step number in the order they were appended; I did not read a clock while writing them.
Earlier drafts carried HH:MM times that I had estimated, not read; they were replaced by step numbers before this file was sent.

## RETRACTED FINDINGS
- SUPERSEDED 2026-11-26 step 11: the prior file's VOID "next block" T2 row is replaced by T2b below (block-level nulls): FALSIFY as registered, H 4 vs 8.67 / 7.42.
- CONTRADICTED 2026-11-26 step 03 (an assumption in the brief, not a prior finding): "labels normally sit to the right of their fragment".
  On the overlays the side changes by row; on f99r labels sit above-left, on f89v2/f100v at the base. A nearest-label join is wrong in 21/89 decidable sampled links.
- No finding of this session has been retracted. The prior file's RETRACTED block is carried by reference (0fa1734c-RUNNING_RESULTS_slots_review.md, read in full).

## BLINDING DISCLOSURE (written before any work)
The uploaded concordance CSV (with label_zl) and the prior results file were in context at session start, before Stage 2.
The prior file names repeated labels with their fragment numbers (otoldy 25/79/95; okol 8/177/218; okary 68/76; okeody 234/235;
oky 70/84; otaly 5/99; otoky 4/102) and by-eye notes on them. So Stage 2 cannot be blind to "which fragments share a label" from
memory. Mitigation: Stage 2 crops are graded under random pair codes with no fragment number, page or position shown; the
code->fragment key is written to disk and only joined after grades are saved and hashed.

## INPUT CHECKS 2026-11-26 step 01
- Yale info.json widths (fetched today): 1037112 2714; 1006233 9078; 1006234 4204; 1006235 7796; 1006246 2702; 1006247 2802;
  1006248 2676; 1006249 7486; 1006250 2698; 1006251 8176; 1006252 2981; 1006253 2838.
- cmp_vms_pharma_reference_objects_v01: 421 rows pulled by REST (sha256 58e61588...b50d). Classes: container 49, leaf 167, root 158,
  stem 19, flower 11, whole_plant 17. label_text column = class name (0 rows equal any label_zl), so it does not leak labels.
- Concordance: uploaded CSV vs live table, per-row md5 of (page, locus, fragment, alt, method, row, box): 245 of 245 non-container
  rows identical; the 40 container rows differ only because the live table fills vnu_row. Live table: 285 rows, 194 Lf, 40 Lc, 51 no_label,
  239 distinct fragments, 211 label boxes (verified; prior file said 211 of 234).
- Existing pharma-to-pharma similarity: cmp_vms_hph_motif_object_retrieval_v01 (pharma-pharma rows: root 3,154, plant 3,456) and
  cmp_vms_hph_spine_descriptions_v01 (372 pharma objects with text embeddings = all non-container objects). No new similarity table made.

## STAGE 1 METHOD, fixed 2026-11-26 step 02 before running (s1_join.py). Geometry only; no label text used.
Looking done to set this (page overlays, 1000-px boxes on 2000-px Yale images): f99r, f89v1+f90r, f88v+f89r, f100v+f101r, f101v1+f102r, f89v2.
What the overlays showed, which changes the brief's assumption:
 - Labels do NOT normally sit to the right of their fragment on these pages. On f99r, f88v-f89r and f89v1 they sit ABOVE the drawing
   (over its top edge); on f89v2 and f100v they sit at the BASE of the drawing. So distance is measured without a side preference.
 - The object inventory is uneven: several boxes cover two or more drawings (e.g. f99r row 1 obj 3), and some manual boxes cover
   most of a page (161:25 leaf 665x1273; 175:29 stem 603x635; 180:50-54). Duplicate boxes exist (162:63 = 162:70, 162:68 ~ 162:74, 162:69 ~ 162:75).
 - Spreads hold non-pharma pages: 1006235 right part = f90r (herbal); 1006249 right part = f101r (29 unlabelled fragments).
Rules:
 1. Page of an object = by box-centre x (1000-px units): 162 f88v <310, f89r1 310-580, f89r2 >=580; 163 f89v2 if >=40 (else off-page);
    164 f89v1 <385, else f90r (herbal, excluded); 178 f100v <390, else f101r; 180 f101v <350, f102r1 350-640, f102r2 >=640; others whole image.
 2. Scale object box to Yale px by width/1000. Candidates = fragment-label loci (Lf) with a box on the same page (ambiguous loci kept, flagged).
 3. Row condition (stands in for "same row", since objects carry no row): label centre y within [obj top - 300 px, obj bottom + 300 px].
 4. d = distance from label-box centre to the object box (0 if inside). Best = smallest d among candidates meeting rule 3.
 5. Flags: HIGH d<=150 and runner-up d - best d >=150; MED d<=400; LOW d<=800; NONE (no label within reach) d>800 or no candidate.
    SPAN if >=2 label centres fall inside the object box or box width > 45% of page width. AMBIG if the best locus has no single fragment.
 6. Sensitivity: reach 400 / 800 / 1200 and row margin 150 / 300 / 600 reported.
 7. Eye check: every object linked to a label in the first concordance row of each page, plus every SPAN / LOW / AMBIG / NONE case on
    pharma pages. Graded on link-line overlays: RIGHT / WRONG / UNCLEAR. Error rate = WRONG / (RIGHT + WRONG).

## RESULT 2026-11-26 step 03 — STAGE 1 join (s1_join.py, s1_eyerate.py, s1_final.py)
Table: public.vms_pharma_object_fragment_join_v01 (new; 372 rows; core columns loaded through SQL; 212 linked rows / 152 fragments,
matches the local file). Full columns in out/stage1_object_fragment_join_v01.csv sha256 fbdc3f35c8bcffc8d5505031564e199df9a5ac97a06426376ee648f06a53de5d.
Eye grades: out/s1_eyecheck_grades.csv. Disagreement list: out/stage1_disagreements.csv.
Objects: 372 non-container. Off the labelled pharma pages: 32 on f101r (no labels), 13 on f90r (herbal), 1 off-page -> 326 on labelled pharma pages.
Geometric join at the registered settings (row margin 300 px, reach 800 px): HIGH 78, MED 180, LOW 12, NONE 56; SPAN 35; AMBIG 4.
Sensitivity (NONE count over all 372): margin 150/300/600 x reach 400/800/1200 -> NONE 88-128; HIGH 78-82 throughout.
EYE CHECK, registered sample (first concordance row of each page + every SPAN/LOW/AMBIG/NONE): 209 objects, all graded.
| grade (sample) | n |
| RIGHT (geometric link correct) | 68 |
| WRONG | 21 |
| UNCLEAR (box spans two plants, or plant boundary not decidable) | 39 |
| RIGHT_NONE (no label, correctly; unlabelled fragment or f101r) | 59 |
| NONE_UNBOXED (label exists but has no box) | 19 |
| MISSED (labelled fragment, join said NONE) | 3 |
Error rate of linked, decidable links: 21 / 89 = 23.6% (95% Wilson interval 16-33%). Counting the 32 linked UNCLEAR as wrong: 53 / 121 = 43.8%.
By confidence over everything graded (sample + extra rows): HIGH 63 right / 2 wrong / 13 unclear (3.1% of decided);
MED 93 / 45 / 42 (32.6%); LOW 2 / 6 / 4.
Why the join fails (seen on the overlays, not counted): label side varies BY ROW on the same page (above-left on f99r rows 1-4;
right-of on f88v row 3 and f89r1/f89r2; at the base on f89v2, f100v, f102v2 row 2); and many object boxes cover two plants.
Decision taken (deviation from the brief, stated here): the automatic link is NOT used downstream. fragment_final = the eye-confirmed
link (158 objects) or the eye-corrected fragment (49) or eye-added (5); MED/LOW links that were not graded or UNCLEAR are dropped.
Eye "truth" rests on Petersen left-to-right numbering plus the concordance's own label->fragment assignment, so it is NOT
independent of the concordance; it only checks geometry against that.
Fragment coverage after Stage 1: 152 fragments with >=1 drawn object (151 of the 188 labelled fragments). 37 labelled fragments
have no object: 8 on the f101v2 panel (labels unboxed), and the rest where the only box spans two plants (listed in s1_final output).
Part shown per fragment: root only 70, leaf+root 38, leaf only 35, root+whole_plant 3, stem only 3, other 3. 48 fragments have >1 object.
Concordance disagreements (CHECKS item): 53 WRONG geometric links; the label they hit was assigned by zl_number 24, order_gate 16,
panel_count 7, by_eye 6. Listed in out/stage1_disagreements.csv. None fixed in the concordance. In none of the 53 did I find the
concordance's own label->fragment assignment to be the error; the errors are the geometric join's. NOT independently tested.
Crops viewed: 23 page-region overlays (Yale IIIF, 1500 px wide over 2,700-3,800 Yale px, i.e. 0.39-0.56 of native) plus the 12 whole-page overlays at 2000 px.
P(survives 60d): "geometric nearest-label join is wrong for about a quarter of decidable MED links" 85%; "HIGH links are reliable (<5% wrong)" 75%.

### What I am uncertain about (Stage 1)
- The eye grades are mine alone, made fast at 0.4-0.56 of native resolution; 39 of 209 sample objects were UNCLEAR. A second grader was not used.
- Which drawing a label belongs to depends on a side convention I inferred per row; on f88r row 1, f89v2 row 1 (50/51) and f102v2 row 1 I may have it wrong.
- 162:31 vs 162:33 tag confusion was caught once; others may remain (tags overlap where boxes are duplicated).
- The 41 objects graded outside the registered sample were chosen because I had already looked at those rows; they are not a random sample.

## STAGE 2 GRADING RULES — written 2026-11-26 step 04, BEFORE any pair crop is viewed
Unit: a pair of drawn objects on two DIFFERENT Petersen fragments (fragment_final from Stage 1), same obj_class (root-root, leaf-leaf).
Each object is shown cut out by its own stored mask (mask_url) on white, so label ink outside the mask is not shown; no page, row,
fragment number or label is shown. Pairs carry random 4-letter codes; the code->object key is written to disk unopened until grades are saved.
Grades (per the brief):
 A = same drawing in all visible parts (shape, branching/lobing pattern, swellings, colour scheme all match; differences only in scale/pose).
 B = same in one part, others absent or unclear (e.g. root matches closely; the other part is cropped, faint, or not in the mask).
 C = generic resemblance only (same broad type, e.g. "fibrous pale root", but no specific shared detail).
 D = (added for completeness) clearly different. D and C are both "not same plant" downstream.
Downstream use: A and B pairs feed T4. Because each grade is made on masked crops of single parts, "all visible parts" in A means
all parts visible in the two masks; plant-level identity across other parts is not checked unless both fragments have the other part.
For fragment pairs where both fragments have both a root and a leaf object, the pair grade is the grade of the more-matching part
plus a note on the other part (B if the other part disagrees or is unclear, A only if both match).
Shortlist (cheap, from existing tables only; no new embedding): candidate pairs ranked by (i) motif reciprocal coverage
(cmp_vms_hph_motif_object_retrieval_v01, pharma-pharma), (ii) cosine of stored description embeddings (cmp_vms_hph_spine_descriptions_v01),
(iii) box aspect-ratio similarity, (iv) mean masked colour distance. Top pairs by mean rank are viewed; the number viewed is recorded.
Random-pair calibration: 12 random cross-fragment same-class pairs are mixed into the sheets (codes indistinguishable) to estimate the
grader's false-A/B rate.

## STAGE 2 GRADES SAVED BLIND 2026-11-26 step 05 (before unsealing the key)
out/s2_grades_blind.csv sha256 4ae537b12cc129c978070353b59950292f4215c21e1203a8b2cbae643c03db3c; key ckpt/s2_key_SEALED.json sha256 9a39ae9f...c4f4 (written before grading, not opened).
112 pairs viewed (100 shortlist + 12 random, mixed): A 0, B 3 (all marked tentative), C 36, D 73.
Viewing scale: masked crops, longest side <= 500 px from Yale, shown at <= 330 px (about 0.1-0.6 of native, varies with object size).

## UNSEALED 2026-11-26 step 06 — Stage 2 key joined (out/stage2_pairs_graded_v01.csv sha256 449807f5...3025)
| code | grade | objects | pages | fragments | source |
| WVRX | B | 161:18 / 162:68 | f88r / f89r2 | 11 / 48 | shortlist |
| XKKM | B | 162:3 / 162:7 | f88v / f88v | 15 / 16 (adjacent in row 1) | shortlist |
| KXRL | B | 176:12 / 176:3 | f99v / f99v | 103 / 95 | RANDOM calibration pair |
Calibration: random pairs 12 -> D 10, C 1, B 1 (8%); shortlist 100 -> D 63, C 35, B 2 (2%).
READING (descriptive): the grader's B rate on random pairs is not lower than on the shortlist. On this sample the B grade does not
separate shortlisted pairs from random ones; it cannot be taken as evidence that any pair is the same plant. No A pair found.
Also: the shortlist features (description-embedding cosine 0.95-1.00 for nearly all pairs; motif coverage) did not concentrate C/B grades:
35% C on shortlist vs 8% on random (descriptive; 12 random pairs only).

## PRE-REGISTRATION 2026-11-26 step 07 — T4 (names written differently each time), t4.py
Pairs: grade A or B from Stage 2 (n = 3; one of them is a random calibration pair, kept because the rule is grade-based).
Label of a fragment = its Lf label(s) in the concordance, ZL form: [a:b] -> a; strip { } ? and separators (, .). If a fragment has two
labels, the pair score is the max over label combinations. Similarity = 1 - Levenshtein / max length (glyph = EVA character).
Statistic S = mean similarity over the pairs. Null: for each pair, a random pair of labelled fragments with the same page relation
(same page -> same page, any page; different pages -> |folio-order distance| equal to the observed one, +-1 page allowed if none), 10,000 draws.
CONFIRM "labels of the same plant are related but not identical": S >= null mean + 2 null SD AND no pair has identical labels.
FALSIFY: S within null mean + 1 SD AND power adequate (see below).
Power, stated now: with n = 3 the test can only pass if the pairs are near-identical; I will report the smallest S that clears 2 SD
and a planted run (replace each pair's second label by the first with k random edits, k = 0..3).
Known before running: the labels of these fragments were in my context at session start (uploaded CSV). I have not compared them.

## RESULT 2026-11-26 step 08 — T4 (t4.py; out/t4_result.txt), scored against the step-07 registration
Inconclusive at this power for the question asked: the three input pairs are not shown to be the same plant (one is a random pair).
| pair (fragments) | labels | similarity |
| 11 / 48 | ofaldo / ytarem | 0.167 |
| 15 / 16 | otar.arody / otokol | 0.333 |
| 103 / 95 | oeeesary / otoldy | 0.250 |
S = 0.250; null (page-distance-matched random fragment pairs, 10,000) mean 0.236, SD 0.097; effect +0.014, ratio 0.15; P(null >= S) = 0.42.
Smallest S that clears 2 SD: 0.43. Power measured by planting: if each pair's labels differed by 0-3 random edits, S cleared 2 SD in 100% of 500 runs at every k.
VERDICT as registered: FALSIFY (S within 1 SD; planted power adequate for <= 3 edits). The labels of these three pairs share no more than
random labels at the same page distance. This says nothing about same-plant labelling in general: n = 3, no A pair, and the B grade did not
separate shortlisted from random pairs. P(survives 60d) "these 3 pairs' labels are no closer than random" 90%; "T4 has been tested" 10% (it has not, in substance).

## PRE-REGISTRATION 2026-11-26 step 09 — T3 (labels mark parts), t3.py
Units: one Lf label per fragment (the first, ZL order) for fragments with fragment_final objects in Stage 1, unambiguous number.
Part category from the linked objects: ROOT (root only), LEAF (leaf only), BOTH (leaf+root); other combinations dropped.
Features (naive partitions; no PGCS features used, so rule 19's extra baseline is not applicable): first glyph, last glyph (EVA characters
after the T4 normalisation; levels with < 5 labels pooled into "other"), length in glyphs.
Statistics: G statistic of part x first glyph; G of part x last glyph; Kruskal-Wallis H of length by part.
Null: permute part labels WITHIN page, 10,000 times (seed 3). z = (obs - null mean) / null SD; p = share of null >= obs.
CONFIRM: any feature with p < 0.0167 (0.05/3) and z >= 2. FALSIFY: all three z < 1 AND planted power >= 80% at f = 0.3.
Otherwise Inconclusive at this power. Power: plant a part-keyed change (last glyph -> 'm' in a fraction f of ROOT labels; f = 0.1, 0.2, 0.3, 0.5;
200 runs, 1,000 permutations each) and report the detection rate on the last-glyph test.
Caveat written now: part category comes from my own Stage 1 eye-links and from an object inventory that misses parts (leaves not boxed, etc.).

## PRE-REGISTRATION 2026-11-26 step 09 — T2b (labels point into the next text block), t2b.py
Text: ZL3b-n.txt from quimqu/voynich-scribe @ 8f5c2c76 (sha256 bf5b6d4a...cafc). On each pharma page, loci in file order; a paragraph block =
maximal run of P-type loci; a label run = maximal run of L-type loci. The NEXT BLOCK of an Lf label = the first paragraph block after its run.
Vords: split on '.' and ','; drop <...> tags, {} kept as content, [a:b] -> a, '?' dropped. Label form: normalised label; HIT if the joined label
equals a vord of the block, or (multi-vord label) any of its vords of >= 3 glyphs does.
Statistic H = number of Lf labels hitting their own next block. Nulls built AT BLOCK LEVEL (fixing the voided earlier attempt):
 N1 = each label against a random OTHER block of the same page; N2 = each label against a random block from another pharma page whose
 token count is within +-30% of its own next block. 10,000 draws each; null mean and SD of the hit count.
CONFIRM: H >= mean + 2 SD under BOTH nulls. FALSIFY: H <= mean + 1 SD under both AND planted power >= 80% when 15% of labels are planted
(label inserted as a vord in its own next block). Otherwise Inconclusive at this power. Power curve at 5/10/15/25% planted.

## RESULT 2026-11-26 step 10 — T3 (t3.py; out/t3_result.json), scored against the step-09 registration
Inconclusive at this power, and power was measured: it is near zero for a part-keyed glyph change.
Units: 142 labelled fragments with Stage 1 objects (ROOT 69, BOTH 38, LEAF 35). Null: part permuted within page, 10,000.
| feature | observed | null mean | null SD | z | p |
| first glyph (G) | 13.55 | 13.77 | 4.51 | -0.05 | 0.49 |
| last glyph (G) | 18.49 | 23.29 | 5.84 | -0.82 | 0.79 |
| length (Kruskal-Wallis H) | 18.86 | 10.96 | 4.12 | +1.92 | 0.043 |
Registered bar (p < 0.0167 and z >= 2) not met by any feature. FALSIFY not reachable: planted power on the last-glyph test is 0% at f = 0.1-0.3
and 26% at f = 0.5 (bar was 80% at 0.3).
Why power is so low (descriptive): part is nearly fixed by page. f99r (20 labels) and f99v (18) are all ROOT; f101v all LEAF; only 35 of 142
labels are in the minority part of their page, so a within-page shuffle has little to move.
Descriptive only: mean label length ROOT 5.72, LEAF 5.71, BOTH 7.58 glyphs; BOTH fragments sit mostly on the quire-15 pages (f88r-f89v1).
P(survives 60d) "T3 as designed cannot separate part from page on these 15 pages" 85%; "length differs by part, page fixed" 25%.

## RESULT 2026-11-26 step 11 — T2b (t2b.py; out/t2b_result.json), scored against the step-09 registration
174 fragment labels with a following paragraph block on 15 pages (35 blocks). H = 4 labels occur in their own next block
(okol f88r; chol/ches f89r1; sheol f89v2; doldam f99v).
| null | mean | SD | (H - mean)/SD | count needed for 2 SD |
| N1 random other block, same page (140 labels have one) | 8.67 | 1.28 | -3.65 | 11.2 |
| N2 size-matched (+-30%) block, other pages | 7.42 | 1.95 | -1.75 | 11.3 |
Power measured: planting 5% of labels into their own next block clears 2 SD under both nulls in 100 of 100 runs.
VERDICT as registered: FALSIFY (H <= mean + 1 SD under both nulls; power adequate). A row's labels do not occur in the text block after
that row more than in other blocks. This replaces the voided prior-session "next block" row (prior file, 08:20).
Not registered, descriptive: H sits 3.65 SD BELOW the same-page null (next blocks are not smaller: 52.8 vs 55.6 tokens on average).
ANALYTICAL CLAIM — needs external test before relying on: "labels are avoided in the block right after their row". No mechanism tested;
4 vs 8.7 hits is a small count and the N1 null is not size-matched.
P(survives 60d) "no excess of labels in the next block" 85%; "a deficit" 30%.

## CHECKS 2026-11-26 step 12
- Stage 1 vs concordance (order_gate / by_eye / panel_count): 53 geometric links were WRONG; the labels they hit were assigned by zl_number 24,
  order_gate 16, panel_count 7, by_eye 6 (out/stage1_disagreements.csv). Neither side was changed. No case found where the eye check pointed to
  an error in the concordance itself; this was not tested independently (eye check used the concordance's own fragment order).
- Fragments with no object after Stage 1: 239 - 152 = 87 of all fragments; 37 of the 188 labelled fragments (8 on the f101v2 panel, whose
  labels have no box; the rest are fragments whose only box spans two plants or whose rows I could not split).
- Crops viewed: 12 whole-page overlays (2000 px on 2,676-9,078 px images); 23 page-region overlays (1500 px across 2,676-3,828 px of Yale, 0.39-0.56 of native);
  112 Stage 2 pair panels (224 masked crops, <= 500 px fetched, shown <= 330 px). No 600-dpi SID scans were used this session.
- Prior-file numbers re-verified against tables: concordance 285 rows, 194 Lf, 40 Lc, 51 no_label, 239 fragments, 211 boxes (all match).
- Tables: one new table (public.vms_pharma_object_fragment_join_v01). Nothing existing altered. vms_pharma_fragmap not used.

## What I am uncertain about (session)
1. Stage 1 truth is my own fast eye grading against the concordance's numbering; a second grader would likely move 10-20 grades.
2. Stage 2 used single-part masked crops at low resolution; a "same plant" judgement that needs root + leaf together was not possible for most
   pairs, and the 8% B rate on random pairs shows my B grade is not informative at this scale.
3. T3's part category inherits the object inventory's gaps (missing leaf boxes on root pages would push fragments into ROOT).
4. T2b depends on ZL file order matching the page layout; checked on f99r, f88v, f102v2 only.
5. Blinding: label_zl was in context from the start (uploaded CSV and prior file). Stage 2 crops hid labels; T3/T4/T2b were computed, not judged.

## PRE-REGISTRATION 2026-11-26 step 13 — Stage 2b: whole-fragment pairing from existing crops (no VLM, no SID), s2b_*.py
Why: Stage 2 graded single parts and the B grade did not beat random pairs (1/12 random vs 2/100 shortlist).
Unit: a FRAGMENT = all its Stage 1 objects (fragment_final), each cut from Yale IIIF with its stored mask (dilated), placed in their
original relative positions on white. Labels outside the masks are not shown; codes only, no page/number.
Step A (nomination, exhaustive): every fragment with >= 1 object shown as a thumbnail on contact sheets grouped by part set
(root-only / leaf-only / root+leaf / other), all codes. I list every pair I think could be the same plant. All C(n,2) pairs are in view.
Step B (grading): nominated pairs + 30 RANDOM fragment pairs (same part set; drawn before nomination, seed 26), shuffled, new codes,
shown side by side at <= 600 px. Grades A/B/C/D as in Stage 2; for root+leaf fragments A needs both parts to match.
CALIBRATION RULE (decides whether Step B grades are usable at all):
 USABLE only if the A+B rate on nominated pairs exceeds the A+B rate on random pairs with one-sided Fisher p < 0.05 AND random A+B <= 10%.
 If not usable: stop; T4 is not run and is reported as "untestable with this grader on these crops".
T4 rerun (only if usable): input = A and B pairs from the NOMINATED set only (random calibration pairs never enter T4); same statistic,
null and planted-power procedure as the step-07 registration; same CONFIRM/FALSIFY rule. Pair labels not read until Step B grades are hashed.

## STAGE 2b GRADES SAVED BLIND 2026-11-26 step 14 (before unsealing)
Nominations (48 pairs, from 8 coded contact sheets covering all 152 fragments): out/s2b_nominations.txt sha256 be4be832...98d3.
Grades: out/s2b_grades_blind.csv sha256 1470e54dfbbb3965048551668c1f65cecc8d252bc1657a07e02fc88c6728e797. 77 pairs (48 nominated + 29 random;
one random draw coincided with a nominated pair and was kept once as nominated). A 0, B 12, C 40, D 25. Shown at <= 440 px per fragment.

## UNSEALED 2026-11-26 step 15 — Stage 2b (out/stage2b_fragment_pairs_graded_v01.csv sha256 3f9f47d4...3fd0)
Calibration rule (step 13): nominated A+B 12/48 = 25%; random A+B 0/29 = 0%; one-sided Fisher p = 0.0019; random rate <= 10%. -> USABLE as registered.
CAVEAT the rule does not remove: I nominated and graded, so "nominated beats random" partly measures my own consistency between the two
looks, not that B means same plant. External grading is the only fix.
B pairs (fragments): 71/103, 88/98, 135/132, 40/124, 87/80, 95/103, 78/94, 15/16, 131/134, 186/187, 71/95, 45/7. No A.
Descriptive: 7 of 12 are on the same page, 4 are neighbours in the same row (15/16, 131/134, 186/187, 132/135 same row 1-2 of f100v);
three pairs form one cluster of ribbed roots (71, 95, 103).
T4 input as registered: the 12 B pairs; 45/7 dropped because fragment 45 has no label -> 11 pairs. Shared fragments (71, 95, 103) make the
pairs non-independent; the null draws independent pairs, so the null SD is if anything too small (bias toward a false CONFIRM).

## RESULT 2026-11-26 step 16 — T4 on Stage 2b pairs (t4b.py; out/t4b_result.txt), scored against steps 07 and 13
11 B pairs with labels. Pair similarities 0.14-0.50; none identical. Highest: dar / darary 0.50 (88/98); dokor / orar 0.40 (186/187).
S = 0.287; null (page-distance-matched random labelled pairs, 10,000) mean 0.237, SD 0.050; effect +0.050, ratio 0.99; P(null >= S) = 0.16.
Inconclusive at this power by the 2x rule; under the registered rule the verdict is FALSIFY (S <= mean + 1 SD), but only by 0.0004 (0.007 SD).
Seed check (5 other null seeds): ratio 0.95-0.98, FALSIFY every time.
Power measured: labels planted as copies with 0-4 random edits clear 2 SD in 100% of 500 runs; 5 edits 92%; 6 edits 62%.
So if same-plant labels were spelled within ~4 edits of each other, this test would have seen it; it did not.
Limits: grader = nominator (see step 15 caveat); non-independent pairs; B is not "same plant", only "same in one part".
P(survives 60d): "labels of these 11 look-alike pairs are no closer than random" 75%; "Stage 2b B pairs are the same plants" 35%.

## STORAGE 2026-11-26 step 17 — what is in Supabase (research-agent, ymaqlcfjmdwncdbjprmw)
- public.vms_pharma_object_fragment_join_v01: all 372 rows, all columns (md5 of id|box|fragment_final matches local file).
- public.vms_pharma_pair_grades_v01: 189 rows = Stage 2 (112) + Stage 2b (77) blind grades with keys (md5 matches local file).
- public.vms_pharma_label_tests_v01: 8 rows, one per test/statistic, with verdicts and P(survives 60d).
- handoff_docs: voynich_pharma_label_pairs_results_20261126 (this file), voynich_slots_cold_review_running_results_20261126 (prior session's
  results file as uploaded), voynich_pharma_label_pairs_brief_20261126 (the brief), voynich_pharma_label_pairs_scripts_20261126 (all scripts).
- NOT recoverable: the prior session's scripts (c1-c5, label_split.py, label_in_herbal.py etc.) and herbal_pages_labelled_from_pharma_v01.csv;
  that session's disk is gone. Supabase Storage upload was refused from this workspace, so images, overlays and the zip are not in Supabase.

## STORAGE 2026-11-26 step 18 — GitHub copy of the whole session
- Repo digitalgoldfisj79/Voynichdecomp, branch experiment/pharma-label-pairs-v1-20261126 (not merged to main),
  folder experiments/pharma_label_pairs_session_2026-11-26.
- Everything in the session folder: results file, brief, scripts, data, checkpoints (incl. unsealed keys), outputs, and all images
  (img 25, ov 50, crops 125, masks 213, frag 304, sheets 14, sheets2b 28). 841 files; SHA256SUMS_all.txt in the folder verified
  with sha256sum -c after copying. Image commit 06cd237; this results-file update is the next commit on the branch.
- Supabase pointer: handoff_docs key voynich_pharma_label_pairs_github_20261126.
- Not on GitHub: the zip (redundant: the loose files are there) and ZL3b-n.txt (third-party; t2b.py names its source commit).
