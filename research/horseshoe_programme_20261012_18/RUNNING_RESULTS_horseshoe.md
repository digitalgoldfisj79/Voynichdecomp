# Horseshoe comparison — running results

## RETRACTION R6 (2026-10-17 ~23:20): cold demolition of PREREG 34-37 (context-free check; key numbers re-verified by me)
(a) PREREG 34/34b "CONFIRMS — his stability is not inflated by text noise": MISLEADING. The registered metric (all 1,225 pairs) is dominated by trivially
    decisive cross-group pairs. Re-verified: same-group pairs (126) decisive his 38.9% vs line-bootstrap 23.0% (ratio 0.59; multinomial 0.86); pairs within 0.05
    of each other in his order (106) his 19.8% vs 8.5% (ratio 0.43; multinomial 0.76). Text noise and his point noise were never combined, so combined
    certainty is lower still. CORRECT STATEMENT: the coarse order (groups/sections) is robust; the fine order WITHIN groups is substantially less certain than his page shows.
(b) PREREG 37 "FALSIFIES — his order disagrees with physical adjacency": WITHDRAWN. The anchor set mixed Pelling's proposed ORIGINAL nesting (vellum flaws) with
    ink transfers between pages facing in the PRESENT binding (e.g. B:10-15~B:11-14 cannot both hold), and the power simulation forced that impossible chain and
    assumed his order = writing order (power ~1.00 was built in). Vellum-flaw pairs alone: mean |rank diff| 3.00 vs null 8.33, permutation p 0.035 (3 pairs; leans
    TOWARD agreement). Status: UNRESOLVED. R5's "the anchor test IS powered on his order" is withdrawn with it.
(c) PREREG 35 secondary ("word information beyond glyph style"): DEMOTED to not established. His positions are a function of the transition-distance matrix, so
    residual signal after a linear TD control is still smoothed transition information (circular in principle). Cold check (not re-run by me): adding 3D/5D MDS
    distance as covariate gives z -2.08 / -2.03. Primary (order predicts vocabulary within stratum, z -9) STANDS but is largely glyph style restated.
(d) PREREG 36 ("no within-scribe replacement"): DEMOTED to "no near-complete replacement detected". Positive control used 20 total 0->100% swaps; cold check
    (not re-run by me) reports 0/8 detections with 50% partial replacements and 5/8 with 5 complete swaps; main z rests on ~9 opposed pairs. Partial respelling untested.
(e) Minor: s50 B argument overwritten to 100 (harmless, 100 was intended); s53 rank ties broken by list order (negligible per cold check); 468 ZL tokens keep markup.
Verified OK by the cold check: s50 truly resamples lines (replicate 1 reproduced exactly); 50-bifolium order aligned across all inputs; TD equals his distance (8e-17);
TTLI maps all 206 pages; anchor IDs correct.

## SUPERSESSION R5 (2026-10-17 ~22:45): PREREG 13 "text cannot resolve individual neighbours" was about OUR gradient only
PREREG 13 assumed rank noise SD 3.8 (our rarefied Fiedler gradient). npcompl33t's order, resampled by whole text lines, has within-scribe-1
rank SD ~1.1 (PREREG 34b), so the physical-anchor test IS powered on his order (power ~1.00). Re-run as PREREG 37: Pelling's anchors do NOT sit
together in his order: V+T mean |rank diff| 6.43 vs null 8.32 (SD 2.08), z -0.91; contact transfers z +0.24 (A:2-7 vs A:3-6 are 15 ranks apart);
vellum flaws z -1.60 (p 0.035, below the z 2 bar). Verdict FALSIFIES "his order agrees with physical adjacency". It does not by itself
falsify chronology: nesting/binding adjacency need not equal writing adjacency.

## RETRACTION R4 (2026-10-15 ~20:45): two claims in my 'explanatory power' view were wrong
(a) Implied generators/ciphers had not been compared with the grammar ('none of those are among the XD1 corpora').
    Misleading: ~24 generators incl. faithful Timm (suppresses exact repeats ~0.76, fails boundary pattern), Schwartz/BootLoops
    (fails edge coupling), MG1-3/joint-forward (fail), Naibbe R-T build (adjacent exact 0.428 vs VMS 0.92-1.24) were all run
    under other programmes. Only SGT12 on non-Naibbe cipher classes was open -> PREREG 17.
(b) 'Space>line-break is direction-mixed in medieval text' (from one CREMMA alpha-64 cell) is superseded by XD1 closeout
    28 Sep: Nuremberg reproduces SPACE>LINE_BREAK strongly; attenuation is NOT VMS-specific.

## RETRACTION R3 (2026-10-14 ~09:55): two POSTED-DRAFT claims downgraded after size-equalised rerun (PREREG 11)
(a) SAME-FACE effect (posted z 2.4, p 0.007, 22/33): with every page rarefied to equal size (pages >= 60 words) it is
    z 1.54, p 0.062, 17/27 -> FAILS the registered survival rule (z >= 2, p < 0.01). Raw distances on the same 27 sheets
    give z 1.94, so the effect leans on 6 sheets with short pages. Status: suggestive, NOT robust. Cut from the post.
(b) MRM "conjugate -0.044, 95% CI excludes 0" was presented against zero. The leaf-permutation null mean of that beta
    is -0.019 (null SD 0.0036), and a within-stratum word-shuffle control (no text signal) still gives conj -0.022 with a
    jackknife CI excluding 0. The CI is not a valid test here. Correct statement: excess over null -0.025, z 7.0
    (size-equalised z 6.4). SHEET > OPENING survives on excess terms (facing: real +0.004 vs shuffled +0.005).
STANDS after size equalisation: conjugate coherence z 6.61 (3.2% closer, 33/35 sheets); PERMANOVA sheet z 5.31.
Negative control: within-stratum word shuffle gives conjugate z 0.62 (raw) / 0.38 (rarefied), face z -0.40 / -0.53,
PERMANOVA z -0.99 -> the RAW conjugate and PERMANOVA figures were not size-inflated.

## RETRACTION R2 (2026-10-14 ~09:10): PREREG 9 "SUPPORTS" verdict withdrawn
The split-leaf replication statistic T_w passes on a NEGATIVE control: control windows with words shuffled across the
whole window (no order, no topic) give T_w z 3.93 (CI_start) and 4.06 (Galen_w2). The test detects sheet SIZE
structure (JS distance depends on sample size; both leaves of a sheet share size), not text structure.
VMS T_w z 5.92 and T_nn z 3.13 are therefore uninterpretable as registered. CVA (PREREG 10) is suspect for the same
reason until it passes the same negative control. Replacement: equal-N rarefied distances (PREREG 10b).

## RETRACTED FINDINGS
- R1 (2026-10-12, own bug, caught same session): s2 reported MDS variance shares 0.5088/0.2096 (text)
  and 0.9449/0.0319 (image). Denominator was the top-4 eigenvalues only. Correct shares (all positive
  eigenvalues, s3): text dim1 0.2105, dim2 0.0867; image (folios >=30 words) dim1 0.9276, dim2 0.0308.
  No verdict depended on the wrong figures.

## DATA WARNING (2026-10-12 ~18:20) — two incompatible section schemes in Voynichdecomp
- Canonical map: voynich_section_map.json (root and Paper/ copies byte-identical; source "Voynich.nu per-page
  descriptions"; 10 labels incl. 'text-only'). Content-based, per page.
- enriched_records.pkl `section` (modal per folio) DISAGREES with the canonical map on 100/223 folios, in contiguous
  blocks: f26-f56 map Herbal-A / ER Herbal-B (62 folios); f57-f66 map Herbal-A, Cosmological or text-only / ER Astronomical;
  f67r1-f68v2 map Astronomical / ER Cosmological; f70r1-2 map Cosmological / ER Zodiac; f87-f96 map Herbal-B / ER
  Pharmaceutical; text-only pages labelled with neighbours in ER. Atlas `section_fixed` follows ER, not the map.
- CONSEQUENCE: every "section" in RESULTS 1-3 above is the ER/Atlas scheme, NOT the canonical map. Results are rerun below
  under the canonical map. Which scheme the repo intends as canonical for enriched_records is unresolved: Ed to decide.

## Context (2026-10-12)
- npcompl33t, voynich.ninja thread 6182 "Proposed Manuscript Chronology": page-level glyph
  transition matrices (within words + within lines, spaces removed), pairwise Jensen-Shannon
  divergence, MDS. 2-D plot is a horseshoe; he reads dim 1 / the fitted curve as production
  chronology (Guttman/Kendall seriation). Transliteration not named. Figures login-gated, not seen.
- Ed: the public Nomic v3 word map (UMAP of DINOv3 word-crop vectors) also looks like a horseshoe.
- Different data (text vs images), unit (page vs word) and method (MDS vs UMAP).

## PRE-REGISTRATION (written 2026-10-12 before any data was pulled or computed)
Data: Atlas v3 vectors (37,886 words, downloaded anonymously from public tiles); crop geometry and
labels from word_iiif_links.csv; line numbers from Atlas v3. Folio = unit of analysis for MDS
(225 folios). Seed 20261012.

Q1 HORSESHOE IN IMAGE VECTORS. Folio centroids of word vectors -> cosine distance -> classical MDS.
  Guttman diagnostic: fit dim2 ~ dim1 (linear) and dim2 ~ dim1 + dim1^2.
  PRESENT if quadratic R2 >= 0.30 AND quadratic R2 >= 3x linear R2. ABSENT if quadratic R2 < 0.10.
  Same test on my replication of the text method (char-level EVA bigrams, see deviations).

Q2 GEOMETRY DRIVES THE IMAGE GRADIENT. Folio-level geometry only: mean log crop width, mean
  log height, mean aspect, mean EVA word length, words per folio.
  CONFIRM geometry-driven if ridge R2 (leave-one-folio-out) of image MDS dim1 from geometry >= 0.40
  or any single |Spearman| >= 0.5. FALSIFY if LOO R2 < 0.10 and all |rho| < 0.2. Else UNRESOLVED.
  Word-level: same Spearman check on the Nomic UMAP arc position (1-D Isomap on UMAP x,y).

Q3 SAME GRADIENT AS THE TEXT HORSESHOE. Spearman rho between image MDS dim1 and text MDS dim1
  across folios (sign-free: report |rho|).
  Null: permute text dim1 WITHIN Davis-Layfield section label (Atlas section_fixed), 2,000 draws,
  so agreement that only reflects "both separate sections" is in the null.
  CONFIRM shared gradient if |rho| >= 0.5 AND within-section permutation p < 0.01 AND
  effect >= 2 null-SD above null mean. FALSIFY if within-section result is < 1 null-SD above null mean.
  Repeat after residualising image dim1 on the Q2 geometry set (geometry-partialled).

Rule 19 naive baseline: the text side already uses naive characters; the image side's naive
baseline IS the Q2 geometry set.

Registered deviations from npcompl33t: (a) EVA characters as glyphs (his glyph inventory unknown);
(b) labels are voynichese.com/TT frame, not necessarily his transliteration; (c) classical MDS on
JSD distances (he may use metric/non-metric MDS); (d) I cannot see his ordering, so I compare
to MY replication of his method, not to his numbers.

## RESULTS (2026-10-12, s1-s4; code in scripts/, outputs results_*.json, folio_mds_coords.csv)
Inputs: 37,886 Atlas v3 vectors (unit norm, all joined), 225 folios.

| Q | Pre-registered criterion | Observed | Verdict |
|---|---|---|---|
| Q1 image (folio MDS) | quad R2 >=0.30 present, <0.10 absent | 0.054 (0.054 with folios >=30 words) | ABSENT |
| Q1 text replication | same | 0.298 | NOT RESOLVED (0.002 short of PRESENT; arch clear by eye) |
| Q2 image dim1 | LOO ridge R2 >=0.40 OR any abs(rho) >=0.5 | R2 0.135; rho(log width) +0.642 | CONFIRM by the rho route only |
| Q2 word map arc (UMAP) | same rho rule | rho log width -0.654, word length -0.566, aspect -0.534 | CONFIRM: map arc = word size |
| Q3 raw | abs(rho)>=0.5, p<.01, >=2 null-SD | rho -0.113; null 0.061 +/- 0.031; z 1.65; p 0.053 | THE METRIC DOES NOT RESOLVE; no shared gradient (abs(rho) far below 0.5) |
| Q3 geometry-partialled | FALSIFY if <1 null-SD | rho -0.161; null 0.146 +/- 0.033; z 0.46 | FALSIFIED |

Pre-registration flaw owned: the "quad R2 >= 3x linear R2" clause is vacuous, because MDS/PCA
dimensions are uncorrelated by construction (linear R2 = 0.000 in both). Only the quad R2 clause did any work.

Descriptive:
- Text dim1 IS the Currier axis: A/B AUC 0.9986; section share of variance (eta2) 0.67. Section order
  along text dim1: Herbal-A < Herbal-B < Pharma < Cosmo < Astro < Rosettes < Zodiac < Stars < Balneo.
  f111r ranks 222/225 (npcompl33t: f111 last in 46/50 refits). Replication is face-valid vs his account.
- Image dim1 is NOT Currier (AUC 0.518). Its extreme end is label/diagram pages (f71v-f72r Zodiac,
  f69v Cosmo, f89r Pharma, robust z -14 to -16): a layout axis.
- Word-level PCA of the vectors (unregistered): strong arch, quad R2 0.696; PC1 rho with log width
  -0.639, word length -0.529; PC2 log width -0.525, word length -0.554. A one-gradient horseshoe whose
  gradient is word size.
- The UMAP has 2 kNN components (11,945 + 55 in a 12k sample): a small detached cluster of tiny crops.

EXPLORATORY (s4, unregistered, quarantined):
- Image dim2 separates Currier: AUC 0.896, null 0.533 +/- 0.026, z 14.2. After residualising on folio
  geometry: AUC 0.538, z 0.19, i.e. gone. Mean crop HEIGHT alone gives AUC 0.902.
- Image dim2 vs text dim1 rho -0.525, but image dim2 is geometry-predictable (LOO R2 0.565), so the
  image-text agreement that exists runs through crop geometry.
- Herbal-only (n=110, Herbal A vs B): geometry-partialled AUC 0.690, null 0.555 +/- 0.040, z 3.35. Something
  beyond the 5 geometry numbers separates Herbal-A from Herbal-B pages. Herbal A/B is also Davis hand 1 vs 2
  and different pages: scribe, parchment and page confounds are all open. NOT a Currier finding.

Interpretation (analytical claim, needs external test before relying on it):
The two horseshoes share a shape, not a cause. Text horseshoe = Currier A->B gradient (his). Nomic word-map
horseshoe = word size. Page-level image vectors show no horseshoe; their main axis is diagram vs running text.

P(survives 60d): "Nomic map arc tracks word size" 90%; "image folio MDS shows no horseshoe" 85%;
"text dim1 = Currier axis in my replication" 95%; "no shared image/text gradient beyond geometry" 75%;
"Herbal-only residual separation z 3.35" 40% (exploratory, confounded).
Rule 20: Q3 should go to a cold demolition chat before anything is built on it.

## PRE-REGISTRATION 2 (2026-10-12 ~17:45, before computing anything on eo777 data)
Source: github.com/eo777/voynich-clusters golf-v21-assignments.csv (302 units, 227 pages, 54 sheets;
ZL transcription per the site). Sheet codes match thread 6182 ("D:25-32"), so page_path_position is
taken to be that author's sequence. Mapping: our f101r1->f101r, f101v2->f101v; 225/225 covered.
Bifolium = his `sheet` column (NOT independently verified against Davis-Layfield; noted).

B1 BIFOLIUM COHERENCE. Pairs of pages on the same sheet but DIFFERENT leaves (conjugate leaves: same
skin, usually far apart in reading order). Statistic: mean pairwise distance of those pairs.
Null: permute sheet labels among pages within section (sizes kept), 2,000 draws, seed 20261012.
Coherent if observed is >=2 null-SD below null mean. Run on: image vectors, text replication,
geometry-only (rule 19 baseline), geometry-residualised image.
PREDICTION written now: images WILL be coherent (P=0.85) because conjugate leaves are one piece of
parchment. Therefore image bifolium coherence CANNOT by itself support "production unit" or "time".
Only coherence that survives geometry AND exceeds what parchment explains would be interesting, and
this design cannot isolate parchment, so any image result here is descriptive.

B2 HIS SEQUENCE vs OUR AXES. Spearman between page_path_position and:
 (a) my text-replication dim1: replication check, PASS if abs(rho) >= 0.70.
 (b) image dim1, (c) image dim2, (d) image dim1 geometry-residualised, (e) image dim2 geometry-residualised.
 (b)-(e): within-section permutation null, 2,000 draws. Bonferroni over 4: AGREE if abs(rho) >= 0.5 AND
 p < 0.0125 AND >= 2 null-SD. NO AGREEMENT if < 1 null-SD. Else not resolved.

## RESULTS 2 (2026-10-12, s5 registered; s6-s8 exploratory bounding)
Data: 225 folios, 53 sheets (his `sheet`), 248 conjugate-leaf pairs (236 same section). His path
position missing for 9 folios (f65r, f67v1, f70v1, f71r, f71v, f72v1, f72v2, f73r, f73v).

| Test | Criterion | Observed | Verdict |
|---|---|---|---|
| B1 image, within-section null | >=2 SD below null | 0.0950 vs 0.1153 +/- 0.0066, z 3.11 | COHERENT (predicted P=0.85) |
| B1 text replication | same | 0.3479 vs 0.3986 +/- 0.0027, z 18.67 | COHERENT |
| B1 geometry only (rule 19) | same | 1.624 vs 1.827 +/- 0.043, z 4.73 | COHERENT |
| B1 image, geometry-residualised | same | 0.610 vs 0.810 +/- 0.022, z 9.15 | COHERENT |
| B2a text d1 vs his path | abs(rho) >= 0.70 | rho 0.933 | PASS: replication matches his sequence |
| B2b image d1 | Bonferroni AGREE rule | rho -0.065; z 1.59 | THE METRIC DOES NOT RESOLVE |
| B2c image d2 | same | rho -0.680; null 0.441 +/- 0.033; z 7.36; p 0.0005 | AGREE (letter of prereg) |
| B2d image d1 geo-resid | same | rho -0.173; z 0.24 | NO AGREEMENT |
| B2e image d2 geo-resid | same | rho -0.136; null -0.024 +/- 0.037; z 4.37 | NOT RESOLVED (abs(rho) < 0.5) |

BOUNDING (exploratory):
- PRE-REGISTRATION FLAW, my own: the B2 page-level nulls are anti-conservative. His path is ~constant within a sheet and
  image vectors are sheet-coherent, so the effective n is ~51 sheets, not 216 pages. Sheet-level re-run (s7, 5,000 perms):
  image d2 rho -0.798, null 0.546 +/- 0.062, z 4.10 (survives); image d2 geo-resid rho -0.170, null -0.063 +/- 0.081,
  z 2.89 (survives marginally, small effect); image d1 z 0.90; text d1 rho 0.938, z 3.71.
- B1 with WITHIN-QUIRE null (s6): text z 17.4 (bifolium-specific, beyond quire); image raw z 1.45 (DOES NOT RESOLVE
  beyond quire); image geo-resid z 10.43; geometry-only z 2.46.
- What carries the image-sequence agreement (s8): crop HEIGHT vs path at sheet level within section rho -0.695,
  z 3.08; at page level within section x Davis scribe rho -0.588 vs null 0.587 +/- 0.013, z 0.09. Crop height's
  link to his sequence is fully accounted for by section + scribe (scribes write at different sizes). Not an
  independent physical clock.

Interpretation (analytical, needs external test):
- His bifolium-agreement observation REPLICATES on my independent rebuild of his text method (different
  transliteration frame, TT vs ZL), and survives a within-quire null (z 17.4). Conjugate leaves are textually
  closer than other sheets in the same quire. P(survives 60d) 85%.
- Images: conjugate-leaf similarity beyond geometry (z 10.4) is what one piece of parchment predicts; it is NOT
  evidence about production order. P 85%.
- Images agree with his sequence mainly via crop height = scribe + section. Small residual beyond geometry
  (sheet-level rho 0.17, z 2.89) is unexplained; parchment/sheet appearance, imaging batch, scribe all open. P 35%.
- Bifolium source is HIS `sheet` column; not cross-checked against Davis-Layfield Table 9.1 this session.

## PRE-REGISTRATION 3 (2026-10-12 ~18:08, before computing): bifolium coherence beyond scribe
Question: is text conjugate-leaf similarity (B1 text, z 17.4 within quire) just "same scribe"?
Pairs: conjugate leaves (same sheet, different leaf), text-replication JS distance (s2_textdist).
Scribe = modal Davis hand per folio (Atlas scribe_davis); Currier = modal Atlas currier.
Primary null (B3a): permute sheet labels among pages within stratum = quire x scribe. 5,000 draws, seed 20261012.
Secondary nulls: (B3b) quire x scribe x Currier; (B3c) section x scribe x Currier.
Effect size: % lower mean distance than null mean.
SURVIVES if >= 2 null-SD below null mean on B3a. COLLAPSES (scribe explains it) if < 1 null-SD on B3a.
Between = NOT RESOLVED. Power check reported: number of conjugate pairs whose stratum holds >= 2 sheets.
Rule 19: same test on image geo-residualised distance for comparison (descriptive; parchment-shared).
Prior: P(survives B3a) = 0.45.

## RESULTS 3 (2026-10-12 ~18:15): bifolium coherence beyond scribe (s9 registered, s10 exploratory bounding)
242/248 conjugate pairs share a Davis scribe; 196/248 share Currier. Informative pairs (stratum has >=2 sheets): 158.

| Test | Observed | Null | z | Effect | Verdict |
|---|---|---|---|---|---|
| B3a text, quire x scribe (PRIMARY) | 0.3479 | 0.3658 +/- 0.0016 | 11.37 | 4.9% closer | SURVIVES |
| B3b text, quire x scribe x Currier | 0.3479 | 0.3658 +/- 0.0015 | 11.65 | 4.9% | survives |
| B3c text, section x scribe x Currier | 0.3479 | 0.3764 +/- 0.0020 | 14.44 | 7.6% | survives |
| image geo-resid, quire x scribe (descriptive) | 0.6101 | 0.7172 +/- 0.0138 | 7.77 | 14.9% | expected (shared parchment) |
Prior P(survives B3a) was 0.45: scored as a miss on the low side.

Bounding (exploratory, all quire x scribe null):
- Per sheet: conjugates closer than same-stratum other-sheet pages in 35/38 sheets (sign test p 3.3e-8). Not a few-sheet effect.
- Central bifolia dropped (16 sheets whose halves face in reading order): z 4.01, 3.5% closer.
- Fixed-size profiles (every page downsampled, 5 reps): 60 words z 11.21 (3.2%); 60 + no central z 7.17 (3.8%);
  100 words z 4.68 (2.5%); 100 + no central z 5.59 (4.7%). Not a page-size artefact.

Interpretation (analytical claim, needs external test):
Conjugate leaves share a text state that is not explained by quire, Davis scribe, Currier language, page size or
reading-order adjacency. Most economical reading: each bifolium's text was written in one batch with the sheet as
the working unit (consistent with writing on loose/unbound bifolia). Effect is SMALL (2.5-7.6% lower distance) but
very consistent (35/38 sheets).
NOT ruled out: shared subject matter per sheet (planned thematic sheets); finer hands or sittings than Davis's five
(a sitting IS a production claim, so this would refine, not kill); transliteration artefacts (unlikely, as
transliterators work page by page in folio order and conjugates are far apart).
Independence: my text features are a rebuild of eo777's method on TT-frame labels; same underlying text, so this
corroborates his observation and adds the scribe/size/adjacency controls, not an independent data source.
P(survives 60d) = 70%. Rule 20: send to a cold demolition chat before building on it. Rule 21: quarantined until
re-derived in a later session.

## PRE-REGISTRATION 4 (2026-10-12 ~18:22, before computing): rerun under the CANONICAL section map
Section = voynich_section_map.json mapping (content-based; 'text-only' kept as its own label). 225/225 folios mapped.
Re-run unchanged, same thresholds as originally registered: Q3 raw + geometry-partialled; B1 (image, text, geometry,
image geo-resid) within section; B2 sheet-level (s7 design) within section; B3c section x scribe x Currier.
NEW B4 (subject-matter alternative): conjugate pairs whose two pages carry DIFFERENT canonical sections. Null: for each
such pair (section Sa, Sb), the mean text distance between non-conjugate pages from Sa x Sb sharing the pair's scribe
(fallback: any scribe if none). Statistic: mean over cross-section conjugate pairs; null by drawing one matched
non-conjugate pair per conjugate pair, 5,000 draws.
CLOSER (coherence beyond subject matter) if >= 2 null-SD below null mean. NOT CLOSER if < 1 null-SD. Report n pairs;
if n < 10, verdict is UNDERPOWERED regardless of z.
Prior: P(B3c still survives) 0.85; P(B4 CLOSER) 0.4; P(B4 underpowered) 0.4.

## RESULTS 4 (2026-10-12 ~18:30): canonical section map (s11 registered; s12 exploratory bounding)
Canonical counts: Herbal-A 113, Stars 23, Balneo 19, Pharma 16, Herbal-B 16, Zodiac 12, text-only 8, Cosmo 8, Astro 7, Rosettes 3.

| Test | Atlas/ER scheme (before) | Canonical map (now) | Verdict change? |
|---|---|---|---|
| Q3 raw (image d1 vs text d1) | rho -0.113, z 1.65 | rho -0.113, null 0.028 +/- 0.035, z 2.42, p 0.0075 | still NOT CONFIRMED (abs(rho) far below 0.5); not falsified either |
| Q3 geometry-partialled | z 0.46 FALSIFIED | z 1.89 | FALSIFIED -> NOT RESOLVED (section scheme matters) |
| B1 text within section | z 18.67 | z 20.54, 14.5% closer | no |
| B1 image within section | z 3.11 | z 3.63 | no |
| B3c text section x scribe x Currier | z 14.44, 7.6% | z 13.94, 7.4% | no: bifolium coherence beyond scribe SURVIVES |
| B3c image (raw) | n/a | z 1.59 | image coherence beyond scribe does not resolve |
| B2 image d2 vs path, sheet level | z 4.10 | z 4.46 | no |
| B2 image d2 geo-resid, sheet level | z 2.89 | z 2.25 | weaker, still >= 2 |
| B4 cross-section conjugates (NEW) | n/a | 27 pairs, 0.392 vs 0.420 +/- 0.007, z 3.92, 6.5% closer | CLOSER by the letter of the prereg |

B4 BOUNDING (exploratory), which DOWNGRADES the B4 verdict:
- The 27 pairs come from only 7 sheets (A:1-8, H:57-66, H:58-65, I:67-68, J:69-70, M:76-83, N:85-86). My registered null
  treated pairs as independent, the same pseudo-replication flaw as B2, my own error again.
- Sheet level: 6/7 sheets closer, sign test p 0.0625 (does not reach 0.05).
- Content-vs-content only (dropping 'text-only' pages): 3 sheets, 15 pairs, z 0.57, 2/3 sheets closer. NOTHING.
- The I:67-68 foldout (10 of 27 pairs) runs the WRONG way (conjugates 0.452 vs pool 0.383).
=> The subject-matter alternative to bifolium coherence is NOT ruled out. B4 is UNDERPOWERED in practice.
   Cross-section conjugate sheets are mostly foldouts/text-only, which is a structurally odd subset.

Descriptive: text dim1 section order (canonical): Herbal-A < Herbal-B < Pharma < Astro < Cosmo < text-only < Rosettes <
Zodiac < Stars < Balneo. Section explains less of text dim1 under the canonical map (eta2 0.553) than under the ER scheme
(0.674); image eta2 is unchanged (0.59 / 0.42).

P(survives 60d): bifolium coherence beyond quire x scribe x Currier (text) 70% (unchanged); "conjugates closer across
content boundaries" 25%; "section scheme changes Q3-geo verdict" 90% (it is a fact of these runs).
OPEN DECISION FOR ED: enriched_records `section` vs voynich_section_map.json, 100/223 folios differ. Any past result
stratified by enriched_records section used the non-map scheme.

## PRE-REGISTRATION 5 (2026-10-12 ~18:35, before any code for these tests): five under-exploited methods
Common inputs: 225 folios; text = my rebuild of eo777's glyph-transition profiles (TT-frame labels); section = CANONICAL
map; scribe = modal Davis hand; sheet/leaf = eo777 `sheet` + folio number. Seed 20261012.
PERMUTATION UNIT FOR ALL CODICOLOGICAL NULLS = THE LEAF (recto+verso move together), shuffled among sheets within
quire x scribe. This corrects the page-level pseudo-replication found twice today.
Every method is implemented from scratch (no R in this workspace) and must pass a known-answer GATE first; a failed
gate voids that test.

T1 VARIANCE PARTITIONING. Question: is there a sheet-level component beyond leaf, quire, scribe, section, Currier?
 T1a Mixed model (REML), outcomes text MDS dims 1-3; variance components Currier, scribe, section, quire, sheet, leaf.
     Sheet component tested by likelihood ratio (with vs without sheet), chi-bar-square p (0.5*chi2_0 + 0.5*chi2_1).
     PRESENT if p < 0.01 on dim1 OR dim2. ABSENT if p > 0.10 on all three.
     GATE: on simulated data with known variances (sheet share 10%), recover sheet share within +/-5 points and LRT p<0.01.
 T1b PERMANOVA on the full text JS distance matrix. Sequential terms Currier, scribe, section, quire, sheet; statistic =
     SS_sheet; null = leaf permutation (2,000). Effect = partial R2 of sheet. PRESENT if p < 0.01; ABSENT if p > 0.10.
     GATE: on Euclidean distances with one factor, pseudo-F equals classical one-way ANOVA F (tolerance 1e-8).
 Prior P(T1 sheet PRESENT) = 0.70.

T2 REGRESSION ON DISTANCE MATRICES (MRM). Response: text JS distance for all 25,200 page pairs. Predictors: conjugate
 (same sheet, different leaf), same leaf, facing (adjacent openings in current binding: verso n / recto n+1), log
 book-order gap within quire, same quire, same scribe, same section, same Currier, nuisances abs(delta log n_words) and
 log min n_words. eo777 path EXCLUDED (derived from the text itself, so circular).
 H2a: conjugate coefficient < 0 with leaf-permutation p < 0.01.
 H2b (codicological model comparison): conjugate vs facing standardised coefficients, 95% CI from bootstrap over sheets.
     SHEET > OPENING if conjugate effect is larger and CIs do not overlap; OPENING > SHEET if the reverse; else no call.
     A "sheets written as units" reading predicts SHEET >= OPENING; a "text written in present reading order" reading
     predicts OPENING > SHEET.
 GATE: on synthetic pairs with a planted conjugate effect, recover the sign and the planted beta within 20%.
 Prior P(H2a) = 0.8; P(SHEET > OPENING) = 0.4.

T3 CA / DETRENDED CA + SERIATION. Page x glyph-transition count table (pages >= 30 words).
 Question: is the text structure effectively one-dimensional (the horseshoe = one gradient)?
 ONE-DIMENSIONAL if DCA lambda2/lambda1 < 0.25 AND DCA axis 2 section eta2 < 0.10.
 TWO+ DIMENSIONS if DCA lambda2/lambda1 >= 0.40 OR DCA axis 2 section eta2 >= 0.30. Else NOT RESOLVED.
 Also reported: DCA axis-1 gradient length (SD units; >4 = complete turnover), arch index (quad R2 of axis2 on axis1)
 before (CA) and after (DCA). Spectral (Fiedler) seriation of sheets: rho vs eo777 path (replication) and vs book order.
 GATE: simulated 1-D coenocline (Gaussian responses, 60 sites x 80 features): CA arch index >= 0.5 and DCA reduces it by
 >= 50%.
 Prior P(ONE-DIMENSIONAL) = 0.35.

T4 TREE-LIKENESS / GRADIENT SUFFICIENCY (sheet level, sheets >= 100 words, JS distance).
 Metrics: mean Holland delta score over all quartets, MDS dim1 and dim2 variance shares.
 Model: single A->B mixing gradient. Each sheet = (1-t)pA + t pB, t = sheet's rank-scaled text dim1, pA/pB = pooled
 profiles of the 10% most extreme sheets each end, multinomial counts at each sheet's real token total. 200 sims.
 GRADIENT SUFFICIENT if all three real metrics lie within the sims' 2.5-97.5% envelope.
 NOT SUFFICIENT if any lies > 2 sim-SD outside (direction reported: higher delta = more conflicting/reticulate signal).
 GATE: delta = 0 (+/-1e-9) on an exact additive tree metric; mean delta >= 0.25 on random uniform distances.
 Prior P(GRADIENT SUFFICIENT) = 0.15.

T5 NORMALISED COMPRESSION DISTANCE (no feature choices). Page strings (words ' ', lines '\n'); zlib-9 primary, bz2
 secondary. Conjugate coherence with the quire x scribe LEAF-permutation null (2,000). Also 60-word fixed-size version.
 SURVIVES if >= 2 null-SD on zlib primary. FAILS if < 1 null-SD. GATE: NCD(x,x) < 0.2; NCD of two independent random
 2,000-char strings > 0.9.
 Prior P(SURVIVES) = 0.6.

### GATES (s13, 2026-10-12 ~18:45) before any manuscript run
G1 PERMANOVA F = ANOVA F (9.8525 = 9.8525) PASS | G2 DCA arch 0.996 -> 0.105, axis1 rho 0.99 with true gradient PASS |
G3 delta tree 0.000, random 0.533 PASS | G4 NCD self 0.024, random 1.003 PASS | G5 MRM planted -0.142, recovered -0.149 PASS |
G6 variance components FAIL: share 0.072 vs true 0.10 (within tolerance) but LRT p 0.166 (needed < 0.01).
=> T1a (mixed model) VOID per pre-registration.
Diagnostic s13b (30 sims each, 48 sheets / 192 pages): estimator UNBIASED (mean share 0.090 for true 0.10; 0.199 for 0.20)
but share SD ~0.076 and LRT power at p<0.01 only 23% (10% share) and 57% (20% share). A mixed-model variance-component test
CANNOT resolve a plausible sheet effect at manuscript scale: a power result in its own right.
REGISTERED DEVIATION (written before running T2): H2b CIs by delete-one-sheet JACKKNIFE, not bootstrap over sheets
(cluster bootstrap duplicates sheets and corrupts the conjugate indicator).

## RESULTS 5 (2026-10-12 ~19:05): five methods (s13-s16 registered; s13b, s17 exploratory)
All codicological nulls permute LEAVES within quire x scribe (2,000 draws). Text = rebuild of eo777 profiles, TT frame.

| Test | Pre-registered criterion | Observed | Verdict |
|---|---|---|---|
| T1a mixed-model sheet variance | gate G6 must pass | gate FAILED (LRT power 23% at 10% share, 57% at 20%) | VOID; underpowered at manuscript scale |
| T1b PERMANOVA sheet term | p < .01 present | SS 3.248 vs null 2.514 +/- 0.117, z 6.27, p 0.0005 | PRESENT. Excess over chance = 3.0% of total variance (raw partial R2 0.25 is mostly chance at 29 df) |
| T2 H2a conjugate beta | < 0, p < .01 | -0.044 vs null -0.019 +/- 0.004, z 7.02, p 0.0005; unique R2 0.0016 | CONFIRMED (small) |
| T2 H2b sheet vs opening | jackknife CIs non-overlapping | conjugate -0.044 [-0.067, -0.021]; facing +0.004 [-0.012, 0.019] | SHEET > OPENING |
| T3 dimensionality | lambda2/lambda1 >= 0.40 -> 2+ | DCA lambda 0.117 / 0.073, ratio 0.63; axis-2 section eta2 0.15 | TWO+ DIMENSIONS |
| T4 single A->B gradient | all 3 metrics in sim envelope | delta 0.270 vs 0.208 +/- 0.002 (z 26); MDS1 0.36 vs 0.58 (z -55); MDS2 0.11 vs 0.03 (z 65) | NOT SUFFICIENT |
| T5 NCD conjugate coherence | >= 2 null-SD | zlib z 3.91 (0.6% closer); bz2 z 4.28; 60-word z 4.33 | SURVIVES as registered |
Priors scored: T1 0.70 -> present; H2a 0.8 -> yes; SHEET>OPENING 0.4 -> yes; ONE-DIM 0.35 -> no (2+); GRADIENT 0.15 -> no; T5 0.6 -> yes.

Descriptive: model R2 0.64, dominated by page size (log min words, unique R2 0.254) and same-Currier (0.054). Book-order gap
beta +0.105 (unique R2 0.006): proximity in present order matters at the broader scale, but immediate facing pages are not
special. DCA axis 1 = Currier (eta2 0.78), gradient length 3.8 SD (near-complete turnover). CA on raw counts shows NO arch
(index 0.004), so the MDS horseshoe in my JS rebuild comes from the distance measure, not the counts. Spectral seriation of
52 sheets: rho 0.91 with eo777's path (independent method reproduces his sequence), 0.68 with present book order.

BOUNDING (s17, exploratory):
- SHEET > OPENING robust: facing pairs excluding conjugates (103 pairs) conj -0.044 [-0.067,-0.021] vs facing +0.002
  [-0.012,0.016]; pages >= 60 words: conj -0.047 [-0.079,-0.015] vs facing +0.008 [-0.007,0.022]. Both SHEET > OPENING.
- !! T5 DOES NOT SURVIVE ADJUSTMENT: with NCD as the MRM response, conjugate beta +0.001 [-0.014, 0.015]: no effect once
  size, Currier, section, scribe and order gap are in the model. The raw NCD coherence (z 3.9) is not an independent,
  feature-free corroboration of the bifolium effect. DOWNGRADE T5 to "raw coherence only, explained by covariates".
- T4 against fairer models: section-pooled + noise: delta z 3.2, MDS1 z 8.9, MDS2 z -13 (outside); section x Currier
  pooled + noise: delta z 24, MDS1 z -28, MDS2 z -1.0 (inside). No tested model of section/Currier + sampling noise
  reproduces sheet-level structure; real data carry MORE conflicting (non-tree, non-gradient) signal than either.

Interpretation (analytical, needs external test):
1. Sheets are textual units beyond leaf, quire, scribe, section and Currier (PERMANOVA z 6.3; MRM z 7.0), and the effect is
   attached to the SHEET, not to the present OPENING: conjugate leaves (far apart in reading order) are closer than chance,
   facing pages (adjacent in reading order) are not. This matches what Davis's "independently written sheets" hypothesis
   predicts and runs against "text written continuously in present reading order". P(survives 60d) = 65%.
2. The effect is SMALL: ~3% of text variance; conjugate unique R2 0.16%. Page size and Currier dominate everything.
3. The text is not one gradient: a second DCA axis (ratio 0.63) and reticulate sheet-level structure (delta 0.27) that no
   section/Currier + noise model reproduces. P(survives) 75%.
4. Mixed-model variance components cannot resolve sheet effects at this scale (power 23-57%): a durable methods result. P 90%.
Rule 20: SHEET > OPENING is the claim to send for cold demolition first.

## PRE-REGISTRATION 6 (2026-10-12 ~18:52, before computing): same face vs opposite face of a flat sheet
Geometry: bifolium with leaves a < b (eo777 sheet code). Flat sheet OUTER face = a-recto + b-verso; INNER face =
a-verso + b-recto. Same-face pairs: (ar, bv), (av, br). Cross-face pairs: (ar, br), (av, bv). Each page appears in
exactly one same-face and one cross-face pair, so page size, scribe, Currier and section cancel within each sheet.
Units: standard bifolia only (exactly one page on each of ar, av, br, bv; foldout panels excluded).
F1 PRIMARY: non-central sheets (leaf numbers not consecutive; for central sheets the inner face IS the facing opening,
 which would confound face with reading adjacency). Text JS distance (s14 rebuild).
 Per sheet D = mean(cross-face d) - mean(same-face d). Statistic = mean D over sheets.
 Null: per-sheet sign flip (swapping b's r/v turns same-face pairs into cross-face pairs), exact if <= 20 sheets,
 else 100,000 random flips. Seed 20261012.
 SAME-FACE CLOSER (sheets written flat, face by face) if mean D > 0, one-sided p < 0.01 and >= 2 null-SD.
 NO FACE EFFECT if < 1 null-SD. Else NOT RESOLVED.
 POWER RULE: minimum detectable effect = 2 x null SD. If MDE > 0.019 (the size of the conjugate effect itself,
 0.367 - 0.348 JS) and the result is not significant, the verdict is UNDERPOWERED, not NO FACE EFFECT.
F2 (secondary, confounded): include central sheets.
F3 POSITIVE CONTROL (images): same-face pages share a skin side (hair vs flesh), so image vectors should show
 same-face closer IF my face mapping is right. Prediction P = 0.6. If F3 fails, the face mapping or the image
 instrument is suspect, and F1 is interpreted with that caveat.
F4 rule-19 baseline: crop-geometry distance, descriptive.
Prior P(F1 SAME-FACE CLOSER) = 0.30.

## RESULTS 6 (2026-10-12 ~19:00): same face vs opposite face (s18 registered, s19 exploratory)
40 standard bifolia (7 central); F1 uses the 33 non-central.

| Test | Observed | Verdict |
|---|---|---|
| F1 text JS, non-central (PRIMARY) | mean D 0.0077, null SD 0.0032, z 2.39, p 0.0072, 22/33 sheets; MDE 0.0064 | SAME-FACE CLOSER (narrowly) |
| F2 text, incl. central (confounded) | D 0.0092, z 2.76, p 0.0022, 27/40 | same direction |
| F3 image raw (positive control) | D 0.0076, z 4.31, 30/33 sheets | PASS: face mapping valid; images see skin side |
| F3 image geo-resid | z 3.78, 25/33 | pass |
| F4 crop geometry (rule 19) | z 2.07, 20/33 (includes text-derived word length / count) | geometry also face-coherent |
Prior P(F1) = 0.30 -> occurred.

Bounding (exploratory):
- Skin-side / legibility alternative (same face = same hair/flesh side = same ink legibility = same transliteration
  habits): per-sheet text effect vs image effect rho 0.106 (p 0.56); vs geo-resid image rho 0.332 (p 0.059); vs geometry
  rho 0.079. Text effect at zero image effect (intercept) 0.0053 = ~70% of the total. Skin side does NOT obviously drive
  it, but the geo-resid link (p 0.06) leaves it OPEN.
- Fragility: leave-one-sheet-out min z 2.17 (holds); dropping the 3 largest sheets z 1.68 (fails, harsh asymmetric
  check); median D 0.0092, 10%-trimmed mean 0.0078 (central tendency robust).
- Scale: the same-face effect (0.0077) is ~40% of the whole conjugate effect (0.019).

Interpretation (analytical, needs external test): within a sheet, pages on the same flat face are textually closer than
pages on opposite faces. That is what writing on the flat, unfolded sheet face by face predicts, and what writing
continuously in the bound book does not. MARGINAL (z 2.4, rests partly on the top few sheets), with a live physical
alternative (skin-side legibility shaping transliteration). Decisive check: re-run on a transliteration made from a
different imaging (e.g. multispectral) or by a different transliterator (ZL vs GC vs TT). If the effect is legibility,
it should vary across transliterators; if it is the writing session, it should not.
P(survives 60d) = 45%.

## PRE-REGISTRATION 7 (2026-10-12 ~19:05, before computing): face and sheet effects across transliterations
Source: Voynichdecomp voynich_transcriptions_slim.json (VTC-0.1-slim; 23 transcriber ids, 17 sources).
Set: drop exact duplicates (identical line text on every line); keep transliterations covering >= 25 of the 33
non-central standard bifolia (all four pages present). Profiles rebuilt per transliteration exactly as s14
(characters of that alphabet as glyphs; word block with ^/$ boundaries + line block without spaces).
GATE: TTVE (37,886 tokens = our Atlas word set) must reproduce F1 within +/-20% of D = 0.0077 with the same sign;
otherwise the run is void.
Per transliteration: F1 (same face vs opposite face, sign-flip null) and F5 (conjugate coherence, leaf permutation
within quire x scribe, 2,000 draws) on the folios that transliteration covers.
Hypotheses:
 WRITING SESSION predicts the face effect in essentially every transliteration, across alphabets.
 TRANSLITERATION HABIT / LEGIBILITY predicts inconsistency across transliterators.
 CONSISTENT if D > 0 in >= 80% of transliterations AND median z >= 1.5.
 INCONSISTENT if D > 0 in < 60% of them OR median z < 0.5. Else NOT RESOLVED.
Note: transliterations are NOT independent data (same ink, mostly the same pages); a CONSISTENT result shows
robustness to reading and alphabet, not replication, and cannot exclude a legibility effect that biases all
readers alike.
Prior P(CONSISTENT) = 0.55.

## RESULTS 7 (2026-10-12 ~19:15): across transliterations (s20, registered)
Exact duplicates removed: FTPL=TTPO, PCCA=PCCI, FFSG=FFSG-1, GCGA=GCGI, TTIA=TTII. Skipped for coverage (< 25/33 sheets):
GLLI, FTPL, JSLI (1), JGLI (4), DVML, RZLI. Ran 12. GATE: TTVE D 0.00773 (target 0.0077 +/- 20%) PASS.

| id | tokens | alphabet | F1 sheets | D | z | p | same-face closer | F5 conjugate z | % closer |
|---|---|---|---|---|---|---|---|---|---|
| TTLI | 34,181 | 21 | 32 | 0.0068 | 1.97 | 0.021 | 20/32 | 7.31 | 5.45 |
| PCCA (Currier) | 16,110 | 35 | 25 | 0.0078 | 1.89 | 0.030 | 17/25 | 6.80 | 6.15 |
| FFSG (Friedman) | 32,941 | 23 | 31 | 0.0077 | 2.25 | 0.011 | 22/31 | 7.26 | 5.84 |
| GCGA (Claston v101) | 39,304 | 60 | 32 | 0.0051 | 1.68 | 0.047 | 20/32 | 7.23 | 4.27 |
| TTIA | 37,564 | 22 | 32 | 0.0076 | 2.28 | 0.010 | 22/32 | 7.22 | 5.43 |
| VDRB | 33,830 | 26 | 32 | 0.0063 | 1.93 | 0.026 | 22/32 | 7.32 | 5.18 |
| VDRB-1 | 37,672 | 25 | 32 | 0.0076 | 2.22 | 0.012 | 23/32 | 6.98 | 5.35 |
| TTVE | 37,670 | 22 | 32 | 0.0077 | 2.34 | 0.008 | 22/32 | 7.14 | 5.43 |
| ZLZB | 37,277 | 26 | 32 | 0.0073 | 2.17 | 0.014 | 23/32 | 7.06 | 5.38 |
| ZLZI | 37,262 | 26 | 32 | 0.0073 | 2.16 | 0.015 | 22/32 | 7.18 | 5.41 |
| FFSG-2 | 30,799 | 31 | 31 | 0.0069 | 2.14 | 0.015 | 20/31 | 6.90 | 5.75 |
| RGVN (Reeds/Gillogly) | 19,940 | 48 | 25 | 0.0077 | 1.89 | 0.028 | 19/25 | 6.39 | 5.59 |

Summary: face effect D > 0 in 12/12, median z 2.15 -> VERDICT CONSISTENT (criteria >= 80% and median z >= 1.5).
Conjugate coherence (F5) z 6.4-7.3 in 12/12 (4.3-6.2% closer).
Prior P(CONSISTENT) 0.55 -> occurred.

Interpretation (analytical): both effects are robust to transliterator AND alphabet (EVA, Currier, FSG, Claston v101,
Reeds/Gillogly), with near-identical effect sizes (face D 0.005-0.008). That argues against a reading-habit artefact.
LIMITS: (1) not independent replications (same ink, same pages); individually most face z are ~2, so consistency speaks
to robustness, not to a larger effect. (2) A legibility effect that misleads ALL readers the same way is not excluded.
(3) Bifolium/face assignments are still eo777's and the standard-bifolium subset (32-33 sheets).
P(survives 60d): sheet effect 75% (up from 65%); same-face effect 55% (up from 45%).

## PRE-REGISTRATION 8 (2026-10-14 ~08:25, before computing): is the face effect a reading-order effect?
Context: Step-83 audit (2026-07-10) found within-bifolium decay d1 .591 > d2 .517 > d3 .459 (LSA cosine,
Q13+Q20, z ~2.5, unresolved). In a singulion read ar, av, br, bv: inner face (av,br) = reading distance 1
(middle opening); outer face (ar,bv) = distance 3; cross-face (ar,br),(av,bv) = distance 2.
Data: 33 non-central standard bifolia, text JS (s14 rebuild), per-sheet paired differences, sign-flip null 100k.
 D_inner = cross_mean - d(av,br);  D_outer = cross_mean - d(ar,bv).
H_SEQUENCE (singulion reading order drives it): D_inner > 0 AND D_outer < 0.
H_FACE (each flat face written as a unit): D_inner > 0 AND D_outer > 0.
Verdict on D_outer (the discriminating quantity): FACE if z >= 2; SEQUENCE if z <= -2; else NOT RESOLVED.
Prior: P(FACE) 0.35, P(SEQUENCE) 0.25, P(NOT RESOLVED) 0.40.
### RESULT PREREG 8 (face vs within-sheet sequence), 33 non-central sheets, text JS, sign-flip null 100k
- mean d: inner face 0.3368, cross 0.3452, outer face 0.3383.
- D_inner (cross - inner) = +0.0084, null SD 0.0078, z 1.07 (20/33): the metric does not resolve this.
- D_outer (cross - outer) = +0.0069, null SD 0.0074, z 0.94 (18/33): the metric does not resolve this.
- inner vs outer: +0.0014, null SD 0.0138, z 0.10.
- VERDICT: NOT RESOLVED. Sign of D_outer leans FACE (sequence predicts the opposite) but z<1.
- Tension (not a retraction): Step-83 LSA within-bifolium decay d1>d2>d3 is not reproduced in JS at the
  page-within-sheet level; outer (reading distance 3) is no farther than cross (distance 2).
- P(survives 60d) of "no within-sheet reading-order decay detectable in JS" = 60%.

## PRE-REGISTRATION 9 (2026-10-14 ~08:45, before computing): can the TEXT order sheets? (split-leaf replication + known-order calibration)
Question: Layfield & Davis hold that original order is recoverable from the text. Necessary condition: text carries
sheet-to-sheet ADJACENCY information beyond the group gradient (scribe/Currier/section), and that information is
reproducible from independent text.
Design: 49 VMS sheets with both leaves >= 60 words. Half A = lower-numbered leaf (2 pages), half B = other leaf.
DA, DB = between-sheet distance matrices from A halves only and B halves only (independent text).
Strata (primary) = sheet-modal scribe x Currier x section (12 strata, 44 permutable sheets, 250 within pairs);
secondary = scribe x Currier (5 strata).
Statistics:
 T_w (PRIMARY): Pearson r between DA and DB over WITHIN-stratum pairs; null = permute B labels within strata (10k).
 T_nn: share of sheets (strata >=3) whose nearest within-stratum neighbour is the same under DA and DB; same null.
 T_all (context only): Pearson r all pairs; null = full permutation. Expected large (gradient); not decision-bearing.
Features: PRIMARY glyph/char transition profiles + JS (as s14). SECONDARY word-unigram tf-idf cosine (LSA-like).
CALIBRATION (known order): Circa instans (Latin herbal, 52k words; 2 windows: start, end) and Galen Greek
pharmacology (122k; 3 non-overlapping windows). Text consumed in sequence into 49 singulion sheets, each sheet given
the four page word-counts of one VMS eligible sheet (book order), pseudo-lines drawn from VMS line-length distribution.
Control strata = contiguous blocks with VMS primary stratum sizes [21,6,5,5,3,2,2,1,1,1,1,1], block order shuffled.
Also report true-order recovery: |Spearman| of spectral seriation (full-sheet D) vs truth, overall and in the 21-block.
DECISION RULES:
 Calibration PASS if control T_w z >= 2 in >= 4/5 windows (primary features).
 VMS T_w z >= 2  -> SUPPORTS "text carries within-group ordering information" (Layfield-Davis position possible).
 VMS T_w z < 2 AND calibration PASS -> NOT SUPPORTED: VMS lacks the local signal a sequentially written known text
   shows at the same size. (Does not rule out ordering with physical anchors.)
 Calibration FAIL -> instrument cannot detect order at VMS size; VMS result uninformative either way.
Prior: P(calibration PASS) 0.7; P(VMS T_w z>=2) 0.3.
Naive-baseline rule 19: features are not PGCS-derived; N/A.
### RESULT PREREG 9 (s22_chain_calibration.py -> results_chain_calibration.json)
Calibration (char, T_w z): CI_start 3.34, CI_end 2.91, Galen_w1 2.43, Galen_w2 3.62, Galen_w3 3.33 -> 5/5 >= 2: PASS.
VMS char T_w: r 0.747 vs within-strata null 0.445, null SD 0.051, z 5.92 (p 1e-4). T_nn 0.35 vs 0.156, SD 0.062, z 3.13.
VMS secondary strata (scribe x Currier): T_w z 10.7. VMS word tf-idf: T_w z 4.10, T_nn z 2.43.
REGISTERED VERDICT: SUPPORTS (VMS T_w z >= 2 with calibration PASS).
SELF-FLAG (not a reframing; the verdict stands): T_w is ORDER-FREE. It shows between-sheet distances within groups
are reproducible from independent leaves = the NECESSARY condition only. Any replicable sheet-level property (subject,
layout, size, quire, sub-hand) passes it. The registered decision label "carries ordering information" overstates
what T_w measures. Awaiting bounding (size, quire, feature block) + an order-sensitive test.
Calibration side-finding: even known-order texts recover order only weakly by spectral seriation:
 |rho| overall char 0.21-0.77, word 0.04-0.25; within 21-block char 0.06-0.66.
P(survives 60d) "within-group between-sheet structure replicates across leaves" = 70%. Ordering claim: not made.

## PRE-REGISTRATION 10 (before computing; bounding of #9 + ORDER-SENSITIVE test)
Bounds on VMS T_w (char): B1 partial out half sizes (log-size sum, |diff|, min) from DA and DB; B2 within-stratum
pairs split into same-quire vs different-quire; B3 word-block-only vs line-block-only features; B4 negative control:
control windows with words shuffled across the whole window (expect T_w z ~ 0; z >= 2 = test is broken).
ORDER-SENSITIVE (decision-bearing): cross-validated adjacency (CVA) within the largest group (VMS: scribe1|A|Herbal-A,
21 sheets; controls: 21-block). Build a shortest open path (NN + 2-opt, 20 starts) on DA; score = mean DB over
path-adjacent pairs vs mean DB over all block pairs; and symmetrically path on DB scored on DA; average.
Null: permute labels of the scoring matrix within the block (5k). Also decay: Spearman of DB vs path separation.
 VMS CVA z >= 2 -> local adjacency replicates (chain-like; the text can propose neighbours within a group).
 VMS CVA z < 2 while >= 3/5 controls reach z >= 2 -> VMS structure replicates but is not chain-like at the
   neighbour level; text cannot propose neighbours within the herbal group.
 < 3/5 controls reach z >= 2 -> CVA underpowered at n=21; no call.
Controls also: path-from-full-D adjacency precision vs truth (chance ~ 2/21 = 0.095) and |Spearman|.
Prior: P(VMS CVA z>=2) 0.35.
### RESULT PREREG 10 (s23_bound_chain.py -> results_bound_chain.json)
B1 size-partialled T_w z 5.33; B2 same-quire z 3.16 / diff-quire z 5.44; B3 word-block z 6.38 / line-block z 5.32.
B4 NEGATIVE CONTROL FAILED: shuffled-word controls T_w z 3.93, 4.06 -> see R2.
CVA: VMS Herbal-A scribe1 (n 21) z 7.06, decay rho 0.35; controls z 2.80, 1.04, 0.30, 5.57, 3.13 (3/5 >= 2).
Truth recovery in controls (path on full D): adjacency precision 0.30-0.45 in 21-block (chance 0.095), 0.23-0.35
overall (chance 0.041); |Spearman| 0.01-0.53. All UNINTERPRETABLE until size is removed (R2).

## PRE-REGISTRATION 10b (before computing): equal-N rarefaction
Each half's w-block and l-block counts subsampled without replacement to the same N (N = min over all halves of all
units, per block), JS averaged over 5 draws. Recompute T_w (within-strata), CVA (largest group), truth recovery on:
VMS, 5 controls, 5 word-shuffled controls (negative).
VALIDITY GATE: shuffled controls must give |z| < 2 for T_w and CVA in >= 4/5; else both instruments declared invalid.
If valid: VMS CVA z >= 2 -> neighbour-level structure replicates across leaves; controls define what a sequential
text gives. Truth recovery in real minus shuffled controls = the ordering information text actually carries.
### RESULT PREREG 10b (s24_rarefied.py -> results_rarefied.json), equal N = 569 word-block / 451 line-block transitions per leaf
VALIDITY GATE: shuffled controls T_w |z|<2 in 5/5; CVA |z|<2 in 4/5 (SHUF_Galen_w1 -2.04) -> PASS.
Known-order controls: T_w z -1.00, 0.40, -0.52, 3.06, 2.67 (2/5); CVA z -0.76, -0.03, -0.41, 4.49, 1.86 (1/5).
 -> at ~100 words per leaf, the split-leaf instrument mostly CANNOT see sequence in known texts (Circa instans: never).
Truth recovery (path on full rarefied D), real vs shuffled: adjacency precision overall 0.19/0.13/0.29/0.21/0.33 vs
 0.02/0.02/0.02/0.00/0.06 (chance 0.041); in 21-block 0.20/0.15/0.35/0.30/0.45 vs 0.15/0.00/0.10/0.05/0.10 (chance 0.095).
 -> known texts carry real but weak order information at VMS size: ~1 in 4 true neighbours found.
VMS (Herbal-A, scribe 1, n 21): T_w r 0.647 vs 0.097, null SD 0.094, z 5.87; CVA 0.0328 vs 0.0000, null SD 0.0042, z 7.74;
 decay rho 0.43 (distance in leaf B rises with separation along the path built from leaf A).
READING: VMS within-group structure replicates across leaves FAR more strongly than any sequential known text, and it
 is graded (1-D-like), not just clumps. Descriptive. Whether that axis is WRITING ORDER or another replicable
 continuum (layout, sub-dialect, subject) is NOT identified by text alone.
P(survives 60d): "VMS Herbal-A scribe-1 sheets lie on a replicable within-group gradient" 60%;
 "known texts at VMS size yield ~20-30% neighbour recovery" 65%.
### EXPLORATORY bounding of 10b (s25_herbal_axis.py -> results_herbal_axis.json)
- Word-internal transitions only (no line layout): T_w z 6.11, CVA z 7.87, decay rho 0.46 -> not a line-layout artefact.
- Herbal-A scribe-1 path vs present binding: path-adjacent same-quire 0.15 vs 0.11, null SD 0.07, z 0.57;
  Spearman path vs book position 0.09, null SD 0.22, z 0.39. The metric does not resolve any relation to present order.
- DESCRIPTIVE (circular: path built from these features, p-values not inferential): along the path ^y +0.80, ^o +0.75,
  m$ +0.72, o$ -0.72, ch -0.61. Resembles the earlier ch/y mode split in H1/A (unverified link).
- Status: the gradient is real and replicable; its identity (writing time vs sub-dialect vs subject) is open.
  Needs an external anchor: physical adjacency evidence (contact transfers, stains) for >= a few Herbal-A sheets.

## PRE-REGISTRATION 11 (2026-10-14 ~09:40, before computing): size-equalised rerun of the POSTED bifolium results
Motivation: R2 showed JS distances carry sheet-size structure. Posted headlines used unequalised distances.
Distances: R60 = pages >= 60 words (211), each page rarefied to 334 word-block + 261 line-block transitions, JS mean of
5 draws; R100 = pages >= 100 words (119), 555 + 435. RAW = original s14 distances (all 225 pages) for reference.
Negative control SHUF: words shuffled among pages WITHIN quire x scribe stratum, each page keeps its line lengths
(destroys sheet-specific text, keeps stratum composition and sizes). Run RAW and R60 on SHUF.
Tests (as posted): T_conj conjugate coherence vs leaf-shuffle within quire x scribe (2000) + per-sheet closer count;
T1b PERMANOVA sheet term (1000 leaf perms); T2 MRM conj/facing betas + delete-one-sheet CI (R60 only, + SHUF RAW);
T_face same-face sign-flip on non-central sheets with all four pages present.
GATE: SHUF R60 must give |z| < 2 on T_conj and T_face. If SHUF RAW gives z >= 2, the posted RAW figures are
size-inflated and must be replaced by R60 figures.
SURVIVAL (on R60 real): T_conj z >= 2 with majority of sheets closer; PERMANOVA z >= 2; MRM conj CI excludes 0 and
facing CI includes 0 (SHEET > OPENING); T_face z >= 2 and p < 0.01. Any failure -> that claim is cut from the post.
Prior: P(T_conj survives) 0.85; P(T_face survives) 0.4.
### RESULT PREREG 11 (s26_posted_rarefied.py -> results_posted_rarefied.json)
| run | T_conj z (% closer, sheets) | PERMANOVA z | MRM conj (perm z) / facing | T_face z (p, sheets) |
|---|---|---|---|---|
| REAL RAW | 7.14 (5.3%, 35/38) | 6.27 (s15) | -0.044 (z 7.0) / +0.004 | 2.39 (0.008, 22/33) |
| SHUF RAW (neg) | 0.62 (0.2%, 20/38) | -0.99 | -0.022 (z -1.3) / +0.005 | -0.40 (15/33) |
| REAL R60 | 6.61 (3.2%, 33/35) | 5.31 | -0.044 (z 6.4) / +0.012 | 1.54 (0.062, 17/27) |
| SHUF R60 (neg) | 0.38 (18/35) | - | - | -0.53 (16/27) |
| REAL R100 | 4.20 (2.7%, 16/18) | 4.06 | - | 1.37 (0.094, 7/11) |
GATE PASS. Sheet effect SURVIVES; SHEET > OPENING SURVIVES (on excess-over-null); SAME-FACE FAILS (R3).
P(survives 60d): sheet effect 75% (unchanged); SHEET > OPENING 60%; same face 30% (was 55%).

## POST DRAFT updated 2026-10-14 ~10:05 per R3 (Ed approved diff): same-face section and panels removed; fig1_v2,
fig3_v2 (real vs shuffled control), fig4_transliterations_v2 (sheet panel only); size-equalised figures added.

## PRE-REGISTRATION 12 (before computing): ESTABLISHING the within-group gradient, and what it is
Data: s22 units, leaf halves rarefied to equal N (569 w / 451 l), as 10b.
G1 RELIABILITY (decision-bearing): within a group, 1-D coordinate (Fiedler) from leaf-A distances vs from leaf-B
 distances; statistic |Spearman|; null = permute leaf-B labels (10k). Groups: scribe1|A|Herbal-A (21, primary),
 and scribe x Currier groups with >= 7 sheets (1|A 26, 2|B 11, 3|B 7). Validity: 5 word-shuffled controls must give
 |z| < 2 in >= 4/5 on their 21-block; known-order controls reported for comparison.
 Primary z >= 2 -> gradient ESTABLISHED (a replicable 1-D ordering of Herbal-A scribe-1 sheets). Else not established.
G2 IDENTITY (exploratory, each |Spearman| vs permutation null, z >= 2 counts):
 a) global Currier axis: project each group sheet's profile on (mean Currier-B profile - mean Currier-A profile),
    both means computed from pages OUTSIDE the group; b) eo777 path position (sheet mean of page 'path');
 c) present book position (already z 0.39 on the path; recomputed on the coordinate).
 If (a) z >= 2: gradient is at least partly the A->B axis; direction then follows ONLY under the external hypothesis
 that A precedes B (contested; not tested here). If (a) < 2: an axis separate from A->B.
G3 STABILITY: 200 rarefaction redraws of whole-sheet profiles -> Fiedler rank per sheet (sign-aligned); report mean
 rank and rank SD; the proposed within-group order is reported only if G1 passes.
Prior: P(G1 primary z >= 2) 0.75; P(G2a z >= 2) 0.5.
### RESULT PREREG 12 (s27_gradient.py -> results_gradient.json)
VALIDITY: shuffled controls G1 z 0.07, 0.83, -0.33, -0.70, -1.02 (5/5 |z|<2) PASS. Known-order controls: CI_start -0.64,
 CI_end -0.94, Galen_w1 1.47, Galen_w2 2.70, Galen_w3 2.24 (2/5 reach 2).
G1 PRIMARY Herbal-A scribe 1 (21 sheets): |rho| leaf-A vs leaf-B gradient 0.51 vs null 0.18, null SD 0.13, z 2.52,
 p 0.019 -> GRADIENT ESTABLISHED as registered (modest: ratio 2.5).
 Other groups: scribe1|A (26, mixes sections) |rho| 0.79, z 5.29; scribe2|B (11) 0.71, z 2.47; scribe3|B (7) 0.86, z 2.24.
G2 identity (Herbal-A): Currier A->B axis (computed outside the group) rho +0.53, null SD 0.13, z 2.66;
 eo777 path rho +0.76, z 4.40 (NOT independent: same text, same method family); present book position -0.08, z -0.77.
G3 stability (200 redraws): median rank SD 3.8 of 21 ranks. A-like end: D:27-30, G:49-56, C:20-21, A:1-8, F:42-47;
 B-like end: C:18-23, F:44-45, C:19-22, E:36-37, B:9-16.
READING (descriptive): within Herbal A scribe 1 there is a replicable 1-D gradient, partly the A->B axis, locally
 identical to eo777's seriation, unrelated to the present binding, resolvable to roughly +/- 4 positions.
 It gives DIRECTION only under the external, contested hypothesis that A-like precedes B-like (Timm & Schinner drift).
 Whether it is writing order is untested: needs physical anchors.
P(survives 60d): gradient established 55%; "partly the A->B axis" 50%.

## PRE-REGISTRATION 13 (before computing; PARTIALLY UNBLINDED - see flag): gradient vs published physical links
FLAG: I saw the G3 ranks of Pelling's four vellum-flaw sheets before writing this (noted in the 08:56 reply).
 This test is therefore EXPLORATORY, not confirmatory, whatever it shows (rule 18).
Anchors (Pelling, ciphermysteries ?p=272 and 2009/03/27): 
 V (vellum flaws, proposed original nesting f35-f36-f9-f10-|-f15-f16-f37-f38): adjacent sheet pairs
   E:35-38~E:36-37, E:36-37~B:9-16, B:9-16~B:10-15.
 T (contact transfers between pages facing in the PRESENT binding; partly disputed): A:2-7~A:3-6 (f2v->f3r, judged
   original), A:3-6~A:4-5 (f3v/f4r blue, disputed), B:10-15~B:11-14 (f15r->f14v red), C:19-22~C:20-21 (f19v/f20r blue, disputed).
Statistic: mean |rank difference| on the G3 gradient (mean ranks) over anchor pairs; null = permute ranks over the
 21 sheets (100k). Report V, T and V+T separately. Power: simulate pairs truly adjacent with rank noise SD 3.8.
Expectation stated in advance: underpowered; a null result is uninformative.
### RESULT PREREG 13 (exploratory; results_anchor.json)
V (vellum flaws, 3 pairs): mean |rank diff| 3.62 vs null 5.08, null SD 2.12, z 0.69. The metric does not resolve this.
T (transfers, 4 pairs): 5.03 vs 5.08, SD 1.69, z 0.03. V+T: 4.43 vs 5.09, SD 1.28, z 0.52.
POWER (decisive point): if every anchor pair were truly adjacent, the gradient's rank noise (SD 3.8 of 21) gives
 median z 0.5-0.7 and P(z >= 2) 0-6%. The text gradient is TOO BLURRY to confirm or refute any individual sheet
 adjacency. Text cannot be validated against a handful of physical links; it can only be tested against many.
P(survives 60d) "the gradient cannot resolve individual neighbours at n=21" 80%.

## LITERATURE SCAN 2026-10-14 ~10:40: techniques that could add ORDER information (descriptive; nothing run)
Bottleneck established by PREREG 13: text gradient rank noise SD 3.8/21 -> anchor tests need MANY anchors or a
sharper gradient. Candidates, ranked by information about original order x feasibility:
1 Same-calf grouping by eraser DNA (eZooMS already done on VMS for species, York 2014 ~10 samples; DNA extension:
  Teasdale et al. 2017 York Gospels RSOS 4:170988; non-destructive mtGenome PLOS One 2024 e0299524, 56% success).
  ~14-15 calfskins, ~2 sheets each (Wikipedia [27] / CEUR keynote; UNVERIFIED primary). Gives dozens of same-skin
  pairs = enough anchors. Needs Beinecke sampling. Highest value, not doable by us.
2 Systematic mirror-registration offset search over ALL page pairs (not just current openings). Our f116v
  show-through pipeline (affine field, 7px out-of-sample) is the base. Stolfi report 087: quire-transition offsets
  are post-painting, some displaced/rotated -> they record later stacking states, not writing order.
3 Outer-face exposure wear per sheet from images: tests Davis's singulion storage directly (every sheet's outer face
  worn vs only quire-outer sheets). Zandbergen: hair/flesh "barely distinguishable" in VMS, so a face signal is more
  likely wear/handling than skin side (our image face effect 30/33 - origin unknown).
4 TempSeq (Gervers & Tilahun, arXiv 2311.02578, JDMDH 2024): per-feature smoothness (local-GLM bandwidth) objective
  + simulated annealing; unsupervised; tested on medieval DEEDS. Could sharpen the gradient. Cheap; calibrate on controls.
5 Biostratigraphic sequencing (CONOP / Unitary Associations / Appearance Event Ordination): word types or rare glyph
  n-grams as taxa with first/last appearances. Uses presence/absence of local vocabulary that JS largely ignores.
  Cheap; calibrate on controls. Not known to have been applied to VMS (unverified).
6 Direction: (a) self-citation asymmetry (source forms precede derived) - calibrate on Timm-type generator output in
  repo Paper/s3_all_generators.pkl; (b) scribal learning curve (shape dispersion falling with practice) from glyph
  crops - confounded by scan/page condition.
Not transferable: supervised page-ordering (arXiv 2602.11040: needs known-order training data; their unsupervised
 NN/TSP baselines reach tau < 0.17 even on modern text). Seriation theory (arXiv 2405.08747 minimax rates;
 2510.22452 MDS confidence sets) useful only for formal uncertainty.

## PRE-REGISTRATION 14 (2026-10-14 ~09:10, before computing): do TempSeq-like or incidence (CONOP-like) ordering beat our seriation?
Units: s22 sheets (49), full sheets (both leaves). Equal-size draws: glyph features rarefied as 10b (both halves summed);
word presence from an equal-N word sample per sheet (N = min sheet words over all units). 3 draws per window.
Methods:
 B0 Fiedler on JS (current); B1 shortest open path (NN+2-opt) on JS (current).
 M1 "TempSeq-like" (NOT a faithful reimplementation): order maximising the MEDIAN over features of the lag-1
    autocorrelation of each feature's standardised proportion along the order (robust per-feature smoothness, in the
    spirit of Gervers & Tilahun's median bandwidth). Top-300 glyph features. Simulated annealing (segment reversal + insertion).
 M2 incidence seriation (CONOP LEVEL-like / consecutive-ones): word types present in >= 2 sheets and <= 50% of
    sheets; minimise total gaps inside each type's first-last range. Simulated annealing.
Truth metrics (controls): adjacency precision (share of recovered consecutive pairs truly adjacent) and |Kendall tau|,
 on all 49 and on the 21-block. Negatives: 5 word-shuffled windows.
DECISION: method M beats the better baseline if mean adjacency precision (all 49) exceeds it by >= 0.05 AND it wins in
 >= 4/5 windows; and on shuffled windows its precision stays <= 0.08 (chance 0.041). Otherwise "no improvement".
VMS (no truth): split-leaf reliability in Herbal-A scribe 1 (21): |Kendall tau| between order from leaf A and from
 leaf B, null = random relabelling (10k). Reported for B0, M1, M2. Descriptive unless a method passes on controls.
Prior: P(M1 beats) 0.3; P(M2 beats) 0.35.

## PRE-REGISTRATION 15 (2026-10-14 ~09:25, before fetching images): outer-face exposure wear (singulion storage test)
Data: Yale IIIF page photos (manifest 2002046, 1000 px wide) for all standard bifolia in faces_per_sheet.csv
(4 standard pages each) + f1r, f116v. Photos show the open bound book, so fore-edge and gutter are not cleanly
segmentable: ONLY top and bottom edges are used (page vs dark background), central 50% of page width.
Soiling index per page S = L*(inner band, 20-30% in from top and bottom) - L*(edge bands, 1-5% in from top and
bottom), L* = 60th percentile of luminance in the band (robust to ink/paint). Higher S = darker edges.
Per sheet: D = mean S(outer face: ar, bv) - mean S(inner face: av, br). Each face has one recto and one verso, so
recto/verso lighting and thumbing cancel.
VALIDITY GATE: f1r and f116v (known board/leather discolouring) must both rank in the top 10% of S among fetched pages;
 edge detection must succeed on >= 90% of pages. Else instrument void.
PRIMARY: D over sheets that are NOT the outermost sheet of their present quire (central sheets included), sign-flip
 null (100k). z >= 2 -> outer faces of interior sheets carry extra edge soiling = exposure as single sheets at some
 time (supports singulion storage, or an earlier different nesting). z < 2 -> no evidence of single-sheet exposure.
SECONDARY: D for quire-outermost sheets; per-sheet list. Caveat registered: skin side (hair/flesh) could add a
 uniform colour difference between faces; an EDGE-minus-inner index is used to suppress uniform differences.
Prior: P(gate passes) 0.6; P(primary z >= 2 | gate) 0.4.
### RESULT PREREG 15 (s29_wear.py -> results_wear.json, wear/page_S.csv)
Edge detection 161/161. GATE FAILED: f1r at the 97th percentile of S (pass) but f116v only at the 66th (needs >= 90%).
INSTRUMENT VOID as registered. Not interpreted: interior-sheet D = +2.01, null SD 0.85, z 2.37, 18/29; outermost
 sheets z 1.25; all z 2.65. These numbers carry no verdict because the positive control failed.
Possible reasons (not tested): f116v discolouring may sit at fore-edge/gutter, which this top/bottom-only index
 excludes; photographic lighting differs between pages. Highest-S pages are dominated by quire 20 rectos (f115r,
 f104r, f114r, f111r, f112r) - unexplained, descriptive only.
Next instrument would need proper page segmentation (fore-edge included) from the unbound-leaf geometry; not done.
### RESULT PREREG 14 (s28_order_methods.py -> results_order_methods.json; 3 draws x 5 known-order windows + 5 shuffled)
Adjacency precision, all 49 (chance 0.041): B0 Fiedler 0.069, B1 path 0.214, M1 TempSeq-like 0.121, M2 incidence 0.103.
 Shuffled windows: all methods 0.021-0.069 (valid).
REGISTERED VERDICT: M1 does NOT beat the best baseline (mean -0.093, 0/5 wins); M2 does NOT (mean -0.111, 0/5 wins).
 No improvement in neighbour recovery. Best available: shortest path, ~1 in 5 true neighbours at VMS size.
EXPLORATORY (not registered as decision): global order |Kendall tau| all 49: M2 0.39 vs B0 0.27, B1 0.20, M1 0.11;
 shuffled 0.08-0.09. Incidence seriation recovers the GLOBAL trend best but the LOCAL neighbours worse.
VMS split-leaf reliability (Herbal-A scribe 1, 21): M2 tau 0.47 z 3.54; B0 0.38 z 2.68; B1 z -0.53; M1 z -0.52.
 Known-order controls reach z >= 2 in 2/15 method-window cells (B0 Galen_w3 2.51, M2 Galen_w2 2.39); shuffled 1/15
 (M1 SHUF_Galen_w2 2.32) -> false-positive rate ~1/15 at this threshold.
READING: the VMS has a reproducible GLOBAL within-group gradient (stronger than sequential known texts at equal size);
 no method tried recovers NEIGHBOURS well even in known texts. Text cannot supply sheet-level order; physical anchors must.
P(survives 60d): "text recovers ~20% of neighbours at best at VMS size" 70%; "M2 global gradient replicates in VMS" 55%.

## CONTEXT 2026-10-14 15:10: npcompl33t/eo777 thread 6199 "Voynich words evolve over a linear sequence" + Echo rule
analyzer (github eo777/voynich-rules @42ea839, 13 Oct 2026). Chain 1a>1b>1.5a>2a>3a>3b>3c>4a>4c over 50 pooled bifolia;
~10-20 spelling rules fitted by LP, scheduled per step; app states "direction along the order is the user's working
assumption" and "capacities are bounds, never token ancestry".
DESCRIPTIVE (our scribe labels on their chain pages): 1a, 1b, 1.5a all Davis scribe 1; 2a scribe 1 + 4; 3a scribes 2+3;
3b 2+5; 3c 2+3; 4a 2; 4c 3. The Currier A->B step (2a>3a) coincides with a SCRIBE change, and the B-side stages are
mixed-scribe pools.

## PRE-REGISTRATION 16 (before computing): are interior chain stages intermediates, or mixtures of earlier/later pages?
For each interior stage k (1b, 1.5a, 2a, 3a, 3b, 3c, 4a) on eo777's chain: E = pages of all earlier stages, L = all
later. Word forms with >= 10 tokens in E+k+L. A form "peaks" at k if its rate in k exceeds BOTH E and L with one-sided
binomial p < 0.01 each. N_k = number of peaking forms. Null: permute PAGES (as units) among E, k, L keeping page counts
(= stage k drawn as a mixture of earlier and later pages, page clustering kept), 2000 perms; z = (N_k - mean)/SD.
Interpretation fixed in advance (asymmetric): z < 2 at stage k -> that stage is indistinguishable from a MIXTURE of
earlier and later pages (gradualness there needs no evolution). z >= 2 -> stage k carries concentrated forms beyond
mixing: consistent with genuine intermediates OR stage-specific (scribe/topic) vocabulary - NOT separated by this test.
Sub-test: within scribe 1 only (1a>1b>1.5a>2a, scribe-1 pages), interior 1b and 1.5a.
Descriptive: top peaking forms per stage; whether they appear in Echo's rule network (fit_data 'hits' outputs).
Data: our TT-frame word tokens (eo777 uses ZL3b) - transliteration difference flagged.
Prior: P(z>=2 at 1.5a) 0.6; P(z>=2 at 3a/3b) 0.5.
### RESULT PREREG 16 (s30_chain_mixture.py -> results_chain_mixture.json)
Full chain, peaking forms vs page-mixture null (N_peaks, null mean +/- SD, z):
 1b 22 vs 9.4+/-4.4 z 2.89 | 1.5a 10 vs 8.9+/-4.0 z 0.28 | 2a 30 vs 9.4+/-4.1 z 4.99 | 3a 22 vs 9.1+/-4.1 z 3.14 |
 3b 19 vs 8.6+/-3.9 z 2.69 | 3c 50 vs 8.6+/-4.0 z 10.5 | 4a 45 vs 7.3+/-3.9 z 9.7.
Scribe-1 only chain (1a>1b>1.5a>2a): 1b z 8.04; 1.5a 7 vs 2.3+/-1.6 z 2.86.
VERDICTS as registered: 1.5a ("transitional A") on the full chain is INDISTINGUISHABLE FROM A MIXTURE of earlier and
 later pages (z 0.28); within scribe 1 alone it passes (z 2.86). All other interior stages exceed mixing (intermediates
 OR stage-specific vocabulary; not separated). 3c (Stars) and 4a (Balneo) peaks are section vocabulary (aiin/ar; ol/qol).
EXPLORATORY (not registered; results_chain_rule_enrichment.json): share of peaking forms that are Echo rule outputs:
 2a 15/30 (OR 3.99, Fisher p 0.0004; okeol, cheol, cheody, cheo, cheos...); 3b 15/19 (OR 15.4, p < 1e-4); 1b 0/22;
 1.5a 0/10; 3a 4/22; 3c 8/50; 4a 14/45 (p 0.09). The -eo-/-eol intermediates DO concentrate at 2a, as Echo's
 *ol>*eol>*edy predicts, but 2a = pharma + astro pages (scribes 1 + 4): section vocabulary is not excluded.
 cheol does NOT peak at 1.5a in our TT-frame data (eo777's post locates its first split there; transliteration differs).
P(survives 60d): "1.5a not separable from mixing" 50%; "eo-intermediates concentrate at 2a" 55%.

## RELEASE PACKAGE 2026-10-14 ~16:40: vms_bifolium_order (pkg/vms_bifolium_order.zip)
Self-contained code + inputs (words_by_page TT frame, sheet map, faces, Echo chain pages + rule outputs @42ea839,
Circa instans + Galen word lists). Clean-room run from the zip (SLIM=1) completed with no errors. Numbers differ
slightly from in-session runs because permutation draw order changed; the package outputs are the reference:
 conj z 7.13 (raw) / 6.41 (equal length) / 0.66 & 0.17 (shuffled); PERMANOVA 6.34 / 5.36 / -0.93; MRM conj z 6.85 / 6.28;
 12 translits z 6.51-7.47; calibration and gradient numbers identical to PREREG 14/12; chain 1.5a z 0.28, scribe-1 z 2.99.
Forum reply draft for thread 6199: post/6199/post_6199.txt + five figures.
2026-10-14 16:29: PR #34 merged (merge e8d989c). Package live at
 https://github.com/digitalgoldfisj79/Voynichdecomp/tree/main/research/bifolium_order_tests (HTTP 200 checked).
Archive note: thread 6199 re-crawled 14:24 UTC (6 posts). Archive caps each forum post at 1500 chars
 (9,407 / 74,967 ninja passages at exactly 1500) - full text beyond the cap is not stored. Not fixed.
Prior art to credit: Timm & Schinner 2020 chol>cheol>cheo>chey>chedy gradient (archive voynich_ninja_thread-5752 p57).

## 2026-10-14 — consolidated evidence folder
- voynich_structure_evidence/ committed on branch structure-evidence, PR #35 (https://github.com/digitalgoldfisj79/Voynichdecomp/pull/35). Contents: README mapping 5 questions from thread 6199 -> evidence + status; claims_register_v03 (147); theorems; npcompl regime tables; lexeme candidates; 20 run records; 5 figures. No new analysis, so no new P(60d).
- Correction logged: SG66 random-collapse baseline is 17.2%, not the 7.9% stated in chat earlier.

## 2026-10-15 — certification tables exported; stale status column found
- CORRECTION: the exported claims register certification_status column predates certification and was never updated; all 32 certified v03 lemmas read OPEN_UNCERTIFIED there. The 147-claim breakdown I gave in the draft Super Grammar note (91 open / 41 held / 9 rejected / 5 superseded / 1 retracted) used that stale column. Replaced with certificate counts (v03 run: 32 certified, 2 failed, 8 rejected, 5 superseded, 1 retracted, 55 open).
- Certification tables (certificates, deps, obligations, evidence, release, validation, verifier runs, retractions) exported with row counts matching DB; release manifest SHA matches release note. PR #36. No new analysis.

## 2026-10-15 — synthesis view (descriptive vs explanatory) + XD1 cross-domain lookup
- ANALYTICAL VIEW, not a finding; needs external test. Looked up XD1 (frozen 2026-09-20) valid results: P1 within-token order-2 gain >=2 SD in 15/16 VMS conditions vs 12/16 Gaskell-Bowern human gibberish (both representations); 9/12 CREMMA medieval witnesses resolve P1 order1->2. P3 previous-token morphology positive in 12/12 CREMMA (resolved 4/12). P2 SPACE-minus-LINE_BREAK (alpha 64 only) in CREMMA: 2 resolved positive, 3 resolved negative, 7 unresolved. Lag-1 exact repeats: VMS within-line ratio 1.022 (0.49 SD, metric does not resolve any effect); Nuremberg 0.093 (49.45 SD suppression); CREMMA page-level lag-1 effect negative in 12/12, resolved in 5 of 7 formal witnesses (different unit from VMS within-line - not like-for-like). Generators/verbose ciphers NOT in XD1 corpora list.
- View: certified kernel = high descriptive quality, low specificity; distinctive feature = absent lag-1 suppression (SGT12 candidate, uncertified). P(60d): kernel mostly non-specific 75%; lag-1 non-suppression most distinctive 55%; A->B change not attributable to time vs scribe vs section from text alone 70%.

## 2026-10-15 — PREREG 17: can unit-level substitution of real plaintext produce the SGT12 repetition geometry? (written BEFORE running)
Context check done first (rule 27): natural controls (Nuremberg, ReM 12 docs, CoReMA A1/BS1/SO1/W1, Add.9308 charms), ~24 generators incl. faithful Timm, Schwartz/BootLoops, MG1-3, and Naibbe (R-T build: adjacent exact obs/null 0.428 vs VMS 0.92-1.24) are ALREADY DONE. 3-Oct record names the gap: SGT12 on other cipher classes (letter/syllable verbose, homophonic). This test fills that gap only.
Statistic: SGT12 method — within-line exact-token pairs at lag 1 and lag 2; null = 200 within-line exact-multiset permutations; report obs/null, effect, null SD.
Arms: VMS ZLZI paragraph lines (instrument positive control); Circa instans (Latin) and Galen (Greek) at unit levels WORD, LETTER, NAIBBE-CHUNK (random 1/2-letter chunks, p=.5), SYLLABLE (heuristic vowel-nucleus split); sub-word levels also with k=3 uniform random homophones. Plaintext unit streams are cut into pseudo-lines with VMS paragraph line-length distribution (fixed seed), capped at 40k units.
"Reproduces SGT12" := lag1 obs/null >= 0.90 AND lag2 obs/null >= 1.10 AND lag2 effect >= 2 null SD (mirrors frozen XD1 falsifier).
INSTRUMENT GATES (else VOID): VMS lag1 obs/null in [0.90,1.15] and lag2 >= 1.10 at >= 3 SD; both WORD-level plaintext arms lag1 < 0.90.
FALSIFIES "unit substitution can produce SGT12": no sub-word arm in either language reproduces AND every sub-word arm has lag1 < 0.90.
CONFIRMS it can: any sub-word arm reproduces. Anything else: PARTIAL, no claim.
Scope: memoryless unit substitution (with/without homophones) of two prose plaintexts under pseudo-lineation. Does NOT cover stateful ciphers, nulls/fillers, highly formulaic plaintexts, or physical lineation.

## 2026-10-15 — PREREG 17 RESULT: FALSIFIES 'memoryless unit substitution of real plaintext produces SGT12'
Instrument gates PASS: VMS ZLZI paragraph lines lag1 obs/null 1.018 (effect/null SD +0.41, metric does not resolve any lag-1 departure), lag2 1.226 (+3.76 SD) — reproduces frozen SGT12 (1.024/+0.47; 1.224/+4.16). WORD arms lag1 0.135 (Latin, -17.4 SD) and 0.027 (Greek, -22.1 SD).
All 12 sub-word arms (letter, Naibbe-style chunk, syllable; +/- k=3 homophones; Latin Circa instans and Greek Galen) suppress lag-1: obs/null 0.144-0.539 (-6.8 to -49.3 SD). None reproduces. Closest = Latin CHUNK 0.539 (needs +67% to reach the 0.90 gate). Lag-2 enrichment does occur in some arms (Greek letter 1.275, +12.7 SD) — consistent with XD1 closeout: lag2 alone not diagnostic.
Cross-check: my CHUNK arms (0.28/0.54) bracket the independent Naibbe R-T build value 0.428.
Bound (rule 17): margin to gate is large (max 0.539 vs 0.90). Not bounded: physical lineation (pseudo-lines from VMS length distribution), formulaic plaintexts, other languages, stateful ciphers/nulls/fillers.
P(survives 60d) = 85% for the scoped result; 65% for the generalisation 'no memoryless substitution of natural prose produces SGT12'.
Files: scripts/s31_unit_substitution_sgt12.py; s31_out/results_unit_substitution_sgt12.json; s31_out/arms/*.pkl

## 2026-10-15 — PREREG 18: checks on npcompl33t's reply (written BEFORE running)
Data: ZL words_by_page (bifolium package), eo777 chain stages (data/eo777_chain_pages.csv), Currier/scribe labels from same.
A. Frequency claims (exact forms chy, chey, chedy, chol, chor, cheol; per-1000-token rates by chain stage).
   Claim (i) chey and chedy rise together: CONFIRM if Spearman(stage order, rate) > 0 for both with rho >= .5; else not supported.
   Claim (ii) chy and chey "almost perfectly inversely correlated" along the chain: CONFIRM if Spearman(chy, chey) across stages <= -0.7;
   FALSIFY if > -0.3; between = unresolved. Few stages -> exact permutation p reported; low power acknowledged in advance.
B. Torsten asymmetry ("1a words appear in Bio/Stars, not the reverse"). Metric: token-weighted coverage asymmetry
   cov(1a in X) - cov(X in 1a), where cov(G in H) = share of G tokens whose type occurs in H. Null: pool pages of the two groups,
   reassign whole pages at random keeping each group's token total within 10% (2,000 draws). Also equal-token rarefied version.
   CONFIRM asymmetry beyond size/vocabulary effects if observed >= null mean + 2 null SD in both versions; FALSIFY if < 2 SD in both.
   Note in advance: even a confirmed asymmetry is not direction (a richer later vocabulary or topic subsetting gives the same sign).

## 2026-10-15 — PREREG 18 RESULTS (npcompl33t checks)
Stages kept (>=1500 tokens): 1a,1b,1.5a,2a,3a,3b,3c,4a(Bio),4c(Stars). Exact-form rates per 1000 tokens: chy 17.6 -> 1.1 (falls); chey 11.5,3.4,3.8,6.5,4.2,6.9,7.8,13.6,17.9 (dips then rises; high at 1a too); chedy 0.3 -> 30.4; chol 35.8 -> 2-7; chor 21.3 -> 0.1-0.5; cheol peaks at 2a (10.6).
A(i) CONFIRMED by frozen rule (both rho >= .5): stage~chey rho .63 (exact p .077; Bio-final order .62), stage~chedy rho .88 (p .003; Bio-final .90). chey is non-monotone.
A(ii) UNRESOLVED: chy~chey rho -0.63 (p .076), between -0.7 and -0.3. Not "almost perfectly inversely correlated" on exact forms.
B 1a vs Bio: THE METRIC DOES NOT RESOLVE THIS. Asymmetry 0.090 vs page-permutation null 0.049 (SD 0.023), z 1.80; rarefied z 0.50 -> FALSIFIED beyond size/vocabulary effects.
B 1a vs Stars(3c+4c): asymmetry 0.241 vs null 0.107 (SD 0.021), z 6.40; rarefied 0.117 vs null 0.007 (SD 0.025), z 4.48 -> CONFIRMED beyond size. Not direction (pre-stated). Awaiting bounding: scribe/section-matched comparison (Stars group mixes hands 2/3, sections Stars/Rosettes/text-only).
P(60d): A(i) 70%; B-Bio null 65%; B-Stars asymmetry (descriptive) 65%.
Files: scripts/s32_npcompl_checks.py; s32_out/results_npcompl_checks.json; s32_out/s32.pkl

## 2026-10-15 — PREREG 19: context inheritance, chy->chey (npcompl33t) vs chey->chedy (1-edit/Timm) (written BEFORE outcomes)
Design counts seen before freezing (not outcomes): early = stages 1a,1b,1.5a (7,635 tokens; scribe 1, Currier A); late = 3c,4a,4c (19,414; scribes 2/3, B).
chy 94 early / 25 late; chey 55 / 229; chedy 1 / 421.
Context of an occurrence = previous word + next word within the same line (line edge = EDGE; neighbours outside the top-150 types of early+late pooled = OTHER). Labels (line_num <= 0) excluded.
Statistic S(X->Y) = JSD between context distribution of X in EARLY and of Y in LATE, each rarefied to n=40 occurrences, mean over 300 draws.
Null: S(X->Z) for every LATE type Z != Y with >= 40 occurrences and ending in -y (primary). Sensitivity: (a) next-word context only; (b) Z restricted to ch-/sh- initial, -y final.
A pair is CONTEXT-SUPPORTED if S(X->Y) <= 5th percentile of the null AND z = (null mean - S)/null SD >= 2.
Decision: npcompl33t's route preferred if P1 supported and P2 not; 1-edit route preferred if P2 supported and P1 not; both or neither = the test does not discriminate.
Secondary (synchronic, same rules): within LATE chey vs chedy; within EARLY chy vs chey (controls = EARLY -y types with >= 30 occurrences, rarefaction n=30).
Caveat fixed in advance: early vs late also differ in scribe and section, so a pass shows context continuity across that whole shift, not respelling by one writer.
PREREG 19 POST-HOC INSTRUMENT QUALIFICATION (added AFTER seeing outcomes, labelled as such): positive control the prereg omitted.
For every type W with >= 40 occurrences in BOTH early and late: S(W_early -> W_late) vs controls = all other late types with >= 40 occurrences sharing W's final glyph. Instrument has diachronic power only if self-matches are mostly supported (median z >= 2). If not, P1/P2 negatives are UNINFORMATIVE.
PREREG 19 RESULTS
Pre-registered outcomes: P1 chy->chey THE METRIC DOES NOT RESOLVE THIS: S 0.4825 vs null 0.5011 (SD 0.0376), z +0.49, rank 9/33.
P2 chey->chedy THE METRIC DOES NOT RESOLVE THIS: S 0.4971 vs null 0.4708 (SD 0.0394), z -0.67, rank 25/33. Verdict as frozen: NEITHER_SUPPORTED_NO_DISCRIMINATION.
Sensitivities agree (next-only z +0.38/-0.46; ch/sh controls +0.54/-0.84). Synchronic: late chey~chedy z +0.12 (unresolved); early chy~chey z +2.02, rank 1/11 (passes, barely).
POST-HOC QUALIFICATION FAILS -> whole test VOID / UNINFORMATIVE:
- Same word across periods (12 words with >= 40 occurrences in both): median z 1.11, only 2/12 >= 2 (dar 2.18, y 3.25).
- Same word split in half within one period: early median z 1.17 (2/8 >= 2), late median z 1.47 (13/42 >= 2).
The instrument usually cannot recognise a word as itself at n = 30-40 occurrences, so failing to link chy->chey or chey->chedy says nothing, and the early chy~chey pass (z 2.02) is within what the instrument produces for unrelated pairs often enough not to lean on.
Lesson (rule: build the known-answer control FIRST): I omitted the positive control from PREREG 19; caught post hoc.
Route that could have power: family-level pooling (more occurrences) or the CF-style graph pooling (CF3/CF4), with the self-match control built first.
P(60d) that exact-form context tests at this sample size are underpowered: 85%.
Files: scripts/s33_context_inheritance.py; s33_out/results_context_inheritance.json; s33_out/qualification_self_match.json; s33_out/qualification_synchronic_split.json; s33_out/cmp/*.pkl

## 2026-10-15 — PREREG 20: does "information preserved" by a frequency-balanced edit mapping tell real change from arbitrary pairing? (written BEFORE running)
Context: npcompl33t reports 85-90% information preserved under his stepwise rules vs much less for 1-edit (Timm-style) connections. His exact metric and current rule set are not public in text form (rules live in an app; his forum list is an image). I do NOT reimplement his rules. I test the metric idea itself.
Operationalisation (mine): EARLY = stages 1a,1b,1.5a; LATE = 3c,4a,4c. Types = most frequent types covering 80% of tokens in each period, masses renormalised.
Mapping = min-cost transport of EARLY mass onto LATE mass, edges only between types within Levenshtein distance <= B (B=2 primary, B=1 'Timm-style'); unmatched mass allowed at cost 1, small tie-break cost 0.01 x edit distance. This is the best frequency-balanced edit mapping an analyst could find.
IP = I(W;T)/H(W) over matched early mass (channel from transport flows). Also MATCH = fraction of early mass transported.
Null: break the form<->frequency link in LATE: permute LATE masses among LATE types (vocabulary and frequency spectrum kept), re-solve; 200 permutations.
FALSIFIES 'IP discriminates': |z(IP_real vs null)| < 2 -> high IP is a property of edit-space density and optimisation, not evidence of information-preserving evolution.
CONFIRMS 'IP carries real signal': IP_real >= null mean + 2 null SD.
Secondary (report, no verdict on meaning): MATCH real vs null z (do edits explain frequency shifts better than chance pairings?); IP at B=1 vs B=2 on real data (his 'one-edit destroys information' claim).
Scope: my metric definition; two-point early->late, not his stepwise schedule. A pass would show IP tracks real form-frequency structure, not meaning.
PREREG 20 RESULTS — VERDICT FALSIFIES_IP_DISCRIMINATES
Types: 807 early / 880 late (80% token coverage).
B<=2 edits: IP real 0.8625 vs null (late form<->frequency scrambled) 0.8637, null SD 0.0067, z -0.18 -> THE METRIC DOES NOT RESOLVE THIS. A frequency-balanced <=2-edit mapping preserves ~86% whether or not the late frequencies belong to the forms they belong to. ~85-90% is what the construction delivers.
B<=1 ('one-edit'): IP real 0.8931 vs null 0.9096 (SD 0.0080), z -2.05. One-edit mappings do NOT destroy information under this metric (IP higher than at B<=2).
Secondary MATCH (share of early mass transportable): B2 real 0.932 vs null 0.845 (SD 0.029), z +3.02; B1 0.634 vs 0.504 (SD 0.026), z +5.01 -> real late frequencies sit closer in edit space to early ones than scrambled pairings.
BOUNDING (rule 17), coverage 60% (201/199 types, 100 nulls, B2 only): IP real 0.8565 vs null 0.8670 (SD 0.0089), z -1.18 -> IP conclusion holds. MATCH real 0.870 vs null 0.846 (SD 0.036), z +0.66 -> the MATCH signal does NOT hold at 60% coverage; treat MATCH z 3.0 as fragile.
Caveat: my IP definition (I(W;T)/H(W) on transport flows), not npcompl33t's undisclosed metric; two-point early->late, not his stepwise rules.
P(60d): 'IP of an optimised frequency-balanced edit mapping does not discriminate real from scrambled pairing' 75%; MATCH signal 35%.
Files: scripts/s34_information_preservation.py; s34_out/results_information_preservation.json; s34_out/solves/*.pkl; s34_out_bound_cover60/
PREREG 20 POST-HOC DECOMPOSITION (exploratory, after outcomes): what drives the MATCH excess?
Split matched mass into RETAINED (same form, edit 0) and MOVED (respelled, edit >= 1).
80% coverage, <=2 edits: retained 0.131 vs null 0.071 (SD 0.014), z +4.18; moved 0.801 vs 0.775 (SD 0.033), z +0.80 -> metric does not resolve respelling excess.
80% coverage, <=1 edit: retained 0.184 vs 0.105 (SD 0.012), z +6.59; moved 0.450 vs 0.399 (SD 0.026), z +2.00 (at threshold).
60% coverage, <=2 edits: retained z +0.15, moved z +0.53 -> nothing.
Reading: the MATCH 'hint' is mostly less-common words persisting unchanged at similar frequency (trivial shared vocabulary), not respelling. Only remnant: one-edit respelling z 2.00 at 80% coverage, post hoc, untested elsewhere.
P(60d) that one-edit respelling excess survives a pre-registered replication: 30%.
File: s34_out/match_decomposition.json

## 2026-10-15 — PREREG 21: replication of the one-edit respelling excess (written BEFORE running)
Origin: post-hoc decomposition of PREREG 20 found one-edit RESPELLED mass excess z +2.00 (ZL words_by_page, early 1a-1.5a -> late 3c/4a/4c, 80% coverage).
Method frozen = s34 (min-cost transport, Levenshtein <= 1, unmatched cost 1, tie-break 0.01*edit, 80% token coverage, 200 permutations of late masses), statistic = RESPELLED share (transported mass on edges with edit = 1) real vs null z. RETAINED reported alongside.
Text: paragraph lines only (unit code with P), from voynich_transcriptions_slim.json; stage map = eo777_chain_pages.csv.
PRIMARY replication = Takahashi (TTLI). Tests: (a) early -> late as in PREREG 20; (b) each adjacent stage pair in eo777 order 1a>1b>1.5a>2a>3a>3b>3c>4a>4c (8 pairs), combined by Stouffer z.
CONFIRMS replication: TTLI (a) respelled z >= 2 AND TTLI (b) Stouffer z >= 2.
FALSIFIES: TTLI (a) z < 2 AND TTLI (b) Stouffer z < 2. Otherwise MIXED, no claim.
Secondary: the same on ZL (ZLZI) paragraph lines.
Even a confirmation shows only that growing forms sit one edit from declining forms more than chance pairing; it says nothing about meaning, and section/scribe change is not excluded.
PREREG 21 RESULTS — VERDICT FALSIFIES_REPLICATION
TTLI early->late: THE METRIC DOES NOT RESOLVE THIS. respelled 0.418 vs null 0.407, z +0.42 (retained z +7.64).
TTLI adjacent stages: Stouffer z -3.12 (per pair: 1a>1b -3.99; 1b>1.5a +0.28; 1.5a>2a +1.96; 2a>3a +2.49; 3a>3b -4.54; 3b>3c -1.22; 3c>4a -1.66; 4a>4c -2.15).
ZL paragraph-only (secondary): early->late z +2.21 (reproduces the original hint); adjacent Stouffer -2.21 (pairs: -2.83, +0.77, +1.87, +2.68, -3.36, -1.91, -0.79, -2.67).
Reading: the one-edit respelling excess is ZL-specific and does not hold stage to stage. Negative adjacent z: neighbouring stages keep most mass unchanged (retained z +6 to +22), leaving less to respell than in scrambled pairings -> the respelled share is entangled with retention and is not an independent measure of drift.
EXPLORATORY (post hoc, not a finding): the two transitions across the A->B change (1.5a>2a, 2a>3a) are positive in both transliterations (z +1.87 to +2.68); they coincide with section and scribe change. Would need its own preregistration.
P(60d): 'no general one-edit respelling excess' 75%; the A->B-transition excess 25%.
Files: scripts/s35_respelling_replication.py; s35_out/results_respelling_replication.json; s35_out/solves/

## 2026-10-16 — PREREG 22: does npcompl33t's optimizer find VMS "respelling" beyond what it finds in unchanged text? (written BEFORE running)
Tool: his optimizer (github.com/eo777/voynich-optimizer @963ac31), Chain class used UNCHANGED (identity control, frequency-only LP = certified min mean-TV, joint lambda search).
Support: identity + generic edges only (no named rules, so all arms are treated alike): source word in section t -> target word in t+1 with unit edit distance <= k and section rates within a factor of two (his stated eligibility; 'unclaimed by named allocation' is moot with no named rules). Units: VMS atoms (ch, sh, ckh, cth, cph, cfh single; e/i separate); Latin letters.
Arms (8 sections with his 1b..4a token sizes 2411/1798/3107/2212/1648/7412/4598/6812, start = first section):
 VMS_REAL: ZL paragraph text rebuilt per eo777 stage (1b,1.5a,2a,3a,3b,3c,4c,4a), tokens truncated per stage to his sizes if larger.
 VMS_PAGESHUF (10 replicates): the same VMS pages randomly reassigned to 8 pseudo-sections of the same sizes (same spelling space, no ordering/gradient).
 VULGATE (10 chains): his tokenisation (verified: 612,195 tokens / 46,476 types, archive sha256 6d1acb5f...), consecutive disjoint 8-passage chains, first 10 of his 20.
Primary statistic: relative mismatch reduction R = 1 - TV_LP / TV_identity (ED<=1). Secondary: joint lambda=0.2 (ED<=1) mismatch and all-word identity retained; ED<=2 LP.
CONFIRMS 'routing finds VMS respelling beyond ordinary usage variation': VMS_REAL R > mean + 2 SD of BOTH VMS_PAGESHUF and VULGATE.
FALSIFIES: VMS_REAL R <= mean + 2 SD of VMS_PAGESHUF (density-matched control) -> the reduction reflects the VMS spelling space + sampling, not an ordered change.
Otherwise (beats PAGESHUF but not VULGATE, or vice versa): PARTIAL, no claim.
Also reported: retention at the joint solution for each arm (a real respelling should not need more merging than controls for the same fit).
Limits fixed in advance: my VMS rebuild != his frozen counts exactly; generic support only; page-shuffle keeps pages intact (within-page structure kept).
PREREG 22 RESULTS (frequency-only LP stage) — PRIMARY VERDICT: FALSIFIES
Replication check: my VMS rebuild gives identity TV 0.729 (his frozen 73.07%). Vulgate rebuild matches his corpus exactly (612,195 tokens, 46,476 types, archive sha matches).
ED<=1 (primary): R = 1 - TV_LP/TV_identity. VMS_REAL 0.580; VMS_PAGESHUF 0.596 (SD 0.015, n=10), z -1.09 -> THE METRIC DOES NOT RESOLVE ANY EXCESS; VULGATE 0.106 (SD 0.018), z +26.3.
ED<=2 (secondary): VMS_REAL 0.875; PAGESHUF 0.857 (SD 0.008), z +2.24; VULGATE 0.380 (SD 0.029). But retention at the LP solution is LOWER for real VMS: 0.506 vs PAGESHUF ~0.54 (ED1: 0.686 vs ~0.69) -> the ED2 extra fit is bought with more merging.
Absolute view: identity TV real 0.729 vs page-shuffled ~0.50 (real sections differ more than random page groups: the gradient/section effect). LP residual real 0.306 vs ~0.20 (ED1); 0.091 vs ~0.07 (ED2).
Reading: the large relative reduction on the VMS is a property of its dense spelling space (random page groupings get the same or more), not evidence of an ordered respelling. The Vulgate is not a fair control for this (Latin spelling space is far sparser: R 0.11), the page shuffle is.
Note: VMS_PAGESHUF06 retention metric returned NaN (metric degenerate case); excluded from retention summaries only.
P(60d) ED1 falsification: 75%; ED2 marginal excess real: 30%.
PREREG 22 JOINT STAGE (secondary, his lambda=0.2 search, ED<=1): FALSIFIES, same as primary.
R_joint VMS_REAL 0.579 vs VMS_PAGESHUF 0.596 (SD 0.015), z -1.12 -> THE METRIC DOES NOT RESOLVE ANY EXCESS; VULGATE 0.105 (SD 0.018).
Retention at joint solution: VMS_REAL 0.695 vs PAGESHUF 0.695 (SD 0.009, n=9; replicate 06 NaN), z ~0.
Joint search barely moves from the LP solution in every arm (TV +0.0002 to +0.0008; retention +0.002 to +0.016).
Overall: with his optimizer and eligibility rule, the routing reduction is a property of Voynich spelling density, reproduced by random page groupings; no evidence the ordered sections are linked by respelling beyond that. ED<=2 marginal real excess (z +2.24) comes with lower retention.
Files: scripts/s36_build_vulgate.py; scripts/s37_optimizer_controls.py; s37_out/results_optimizer_controls*.json; s37_out/verdict*.json; s37_out/arms/*.pkl; vulgate/vulgate_tokens.json

## 2026-10-16 — PREREG 23: does the VMS carry the slot-wise periodicity a ROTATING volvelle must leave? (written BEFORE running)
Idea (new): Cham's rings alternate filled/blank cells. If rings advance step-wise between words, each ring's output repeats with the ring's step period, so slot values recur at EVEN lags more than odd lags within a line. Independent draws (no rotation memory) leave no such pattern. This is a sequence signature, so it needs no folio order and no knowledge of ring contents.
Instrument: per naive slot (rule 19: FIRST atom, LAST atom, LAST-2 atoms; also WHOLE word), within-line recurrence excess at lags 1-6 vs within-line exact-multiset permutation null (200 perms). Periodicity index PI = mean z(lags 2,4,6) - mean z(lags 1,3,5); SD of PI from 200 line-bootstrap resamples; report PI / SD.
POSITIVE CONTROL FIRST (synthetic, Cham toy architecture: 2 volvelles x 4 rings, alternating filled/blank cells, word picks a volvelle at random, each ring advances +1 with p=0.7 after use): must give PI/SD >= 3 in at least one slot. NEGATIVE CONTROL: same volvelles, ring positions redrawn at random each word: all slots |PI/SD| < 2. If either control fails -> VOID.
Text arms: VMS ZL paragraph lines (all pages); Vulgate tokens cut into pseudo-lines with the VMS line-length distribution (same naive slots on letters).
CONFIRMS a rotation signature: any VMS slot PI/SD >= 3 AND the same slot in Vulgate < 2.
FALSIFIES (for step-wise rotating alternating-blank rings at the word level): every VMS slot PI/SD < 2.
Otherwise PARTIAL. A confirmation would show periodicity, not volvelles specifically; a falsification does not exclude volvelles read without stepwise rotation.
PREREG 23 RESULT — VOID (positive control failed): rotating toy volvelle max PI/SD 2.42 (< 3); random-position control passed (all |PI/SD| < 2).
Diagnosis: (1) within a line the multiset is fixed, so any short-lag excess forces long-lag deficits; the even-minus-odd index mixes that conservation with periodicity; (2) Vulgate PI is driven by strong lag-1 suppression (z -23.8), not periodicity. Instrument badly designed; no verdict.
EXPLORATORY (post hoc, not a finding): WHOLE-word lag profiles (z, lags 1-6):
 rotating toy volvelle  1.0, 5.7, 0.9, -1.0, -2.0, -2.0  (lag-1 at chance, lag-2 enriched = SGT12 shape)
 random-position toy   -1.5, 1.2, 1.0, 0.9, -0.9, -1.6
 VMS (ZL paragraphs)    0.4, 3.7, 2.5, 1.4, -1.5, -1.4
 Vulgate pseudo-lines -23.8, 1.7, 2.2, 6.1, 5.5, 6.4
A stepwise-rotating alternating-blank volvelle produces the SGT12 geometry without being designed to; no other generator or text in the record does. Needs a properly designed, pre-registered profile-matching test against fitted alternatives.
Files: scripts/s38_volvelle_periodicity.py; s38_out/results_volvelle_periodicity.json; s38_out/arms/*.pkl

## 2026-10-16 — PREREG 24: fitted generator comparison on the within-line repeat profile (written BEFORE running)
Target: within-line repeat PROFILE = obs/expected ratio at lags 1-6 for 4 naive slots (WHOLE word, FIRST atom, LAST atom, LAST2 atoms) = 24 cells. Expected count uses the exact within-line permutation expectation (analytic). VMS ZL paragraph lines, atoms as before.
Split: pages randomly halved (seed 20261016); split A fits on half 1 and tests on half 2, split B the reverse.
Models (each emits a token stream cut into lines with the training line-length distribution):
 M0 INDEPENDENT: words iid from the training unigram. 0 parameters.
 M1 SELF-CITATION (Timm-style family): with prob c copy a word from the previous w tokens (uniform), then with prob m change one random atom to a training-frequency atom; else draw from unigram. Grid c{0,.05,.1,.2,.3,.45} x w{1,2,3,5,8} x m{0,.3,.6}.
 M2 VOLVELLE (Cham-style): V volvelles x 4 rings x 24 cells; ring r holds word part r (part A = first atom, B = middle atoms, C = second-last atom, D = last atom; parts sampled from training part frequencies); even cells always filled, odd cells blank with prob b; each word picks a volvelle at random, emits the 4 cells at the pointers, then each ring advances +1 with prob s. Grid s{.3,.5,.7,.85,1.0} x b{0,.25,.5,.75} x V{1,2,3,5}. 3 parameters each for M1 and M2.
Fit: choose grid point minimising D_train; evaluate on held-out half: D = sum over 24 cells of ((gen - heldout)/SE)^2, SE = line-bootstrap SE of the held-out cell (200 resamples); generated profile = mean of 5 generated samples of held-out size.
CONFIRMS volvelle advantage: in BOTH splits M2 has the lowest held-out D, and D(next best) - D(M2) > 2 x its line-bootstrap SD (paired bootstrap of held-out lines, 200).
FALSIFIES: M2 not lowest in either split, or margin < 2 SD in either split.
Report all D values and the fitted parameters. A confirmation shows a stepwise-rotation generator reproduces the repeat profile better than these alternatives, not that volvelles were used.
PREREG 24 RESULT — VERDICT FALSIFIES_VOLVELLE_ADVANTAGE (by the frozen margin rule), but the volvelle model is best in both splits.
Held-out D (lower = closer; 24 cells, so ~24 would be a fit within noise):
 split A: M2 volvelle 77.7 (s .85, b .75, V 3); M1 self-citation 133.9 (c .05, w 5, m .6); M0 independent 161.0. Margin M1-M2 = 55.0 vs paired-bootstrap SD 36.5 -> 1.5 SD (< 2: fails).
 split B: M2 107.9 (s .85, b .5, V 5); M1 187.8 (c .2, w 5, m .6); M0 200.4. Margin 76.0 vs SD 38.6 -> 1.97 SD by bootstrap mean (point margin 79.9 = 2.07 SD).
No model is adequate (best D 78-108 vs ~24 expected at a noise-level fit).
Whole-word profile (lags 1-6), split A held-out: VMS 1.02 1.21 1.16 1.15 0.91 0.72 | volvelle 0.94 1.12 1.07 1.07 0.89 0.91 | self-citation 1.06 1.19 1.16 1.15 1.17 0.50 | independent ~1.0 flat. Self-citation with a 5-word window also produces the lag-2 bump, so the SGT12 shape is not unique to volvelles.
Reading: stepwise-rotation generator is consistently closest on this profile, but not decisively better than a fitted self-citation model, and neither fits.
P(60d) 'volvelle-type generator fits the repeat profile best among simple generators': 45%; 'decisive advantage': 20%.
Files: scripts/s39_generator_profile_fit.py; s39_out/results_generator_profile_fit.json; s39_out/splits/*.pkl

## 2026-10-16 — PREREG 25: systematic refinement of the volvelle model, with an equally refined rival and a locked test set (written BEFORE running)
Guard against forking paths: pages split at random (seed 20261016) into FIT 50% / DEV 25% / LOCKED TEST 25%. TEST is touched once, at the end.
Score = D over 28 cells, each standardised by line-bootstrap SE of the target: the 24-cell within-line repeat profile (PREREG 24) + 4 vocabulary cells (share of generated tokens attested in the fitting vocabulary; type/token ratio at fixed size; mean word length in atoms; share of 1-atom words). Noise reference: D between the real DEV and real TEST halves' profiles computed the same way.
Refined VOLVELLE family (5 params): s (independent ring step prob), kappa (prob all 4 rings step together), b (blank share of odd cells), V (volvelles), rho (prob of staying on the same volvelle). Ring columns are real training words split into 4 parts, so in-sync rings emit real words and out-of-sync rings build new ones.
Refined SELF-CITATION family (5 params): c (copy prob), w (window), window shape (uniform/geometric), m (one-atom mutation prob), a (prob of rejecting a lag-1 copy).
Same budget for both: 150 random configurations scored FIT->DEV (2 samples each), top 5 re-scored (5 samples), best kept. Final: refit on FIT+DEV, score on TEST (5 samples), paired line-bootstrap (200) for the difference.
CONFIRMS volvelle advantage: D_test(self-citation) - D_test(volvelle) > 2 bootstrap SD.
FALSIFIES: margin <= 2 SD (or self-citation better).
ADEQUACY (separate): a model is adequate if D_test <= the 95th percentile of the real-vs-real noise reference (bootstrap of DEV-vs-TEST real D).
PREREG 25 RESULT — VERDICT FALSIFIES_VOLVELLE_ADVANTAGE (decisively): locked TEST D volvelle 5102.5 vs self-citation 262.9; margin -40.7 SD.
Best DEV configs: volvelle s .999, kappa .23, b .01, V 8, rho .46 (D_dev 4927); self-citation c .40, w 11 geometric, m .54, a .29 (D_dev 105.9).
Where the volvelle fails: vocabulary. Type/token at 5,000 tokens 0.069 vs VMS 0.358; attested share 0.64 vs 0.857. Repeat-profile part alone: volvelle 126.0 vs self-citation 36.6 (24 cells). Whole-word lag-1 ratio 1.58 vs VMS 1.06 (8 small wheels repeat too much).
Self-citation passes the adequacy benchmark (D 262.9 <= real-vs-real p95 1355.1), BUT that benchmark is inflated by vocabulary differences between page sets (real DEV vs TEST D 1159.9), so 'adequate' is weak.
Design limit (mine): 24 cells per ring, no wear/replacement -> a few hundred distinct columns cannot reach VMS vocabulary richness. Cham's model includes ring replacement over time; not modelled. The locked test is now used; any further volvelle refinement needs a fresh test (e.g. Takahashi transliteration or a held-back section).
P(60d) 'fixed small-volvelle generators cannot match VMS vocabulary richness': 80%; says nothing yet about volvelles with replacement/large rings.
Files: scripts/s40_volvelle_refine.py; s40_out/results_volvelle_refine.json; s40_out/stages/*.pkl

## 2026-10-16 — PREREG 26: volvelle WITH ring wear/replacement and variable ring size (written BEFORE running)
Motivation: PREREG 25 volvelle failed on vocabulary (fixed 24-cell rings, no replacement). Cham's mechanism includes ring replacement.
VOLVELLE v3 (7 params): PREREG 25 params (s, kappa, b, V, rho) + N cells per ring (12-240) + r = per-word probability that one ring of the current volvelle wears out and is replaced by a fresh ring (cells = that ring's part of frequency-sampled training words; odd cells blank with prob b).
Rival: SELF-CITATION as in PREREG 25 (5 params; its best FIT->DEV config reused). The volvelle has 2 extra parameters, which biases toward it: a falsification is therefore stronger than a confirmation.
Budget: 150 random volvelle configs FIT->DEV (2 samples), top 5 re-scored (5), best kept. Same 28-cell D.
Test: the PREREG 25 locked TEST pages were scored once before and never used for tuning; reused once here (disclosed). Second, independent-transliteration check: same TEST pages in Takahashi (TTLI), with models refit on TTLI FIT+DEV using the chosen configs.
CONFIRMS volvelle advantage: margin D(self-cite) - D(volvelle) > 2 bootstrap SD on BOTH ZL and TTLI test.
FALSIFIES: margin <= 2 SD on either. Report repeat-only and vocabulary-only parts separately.
PREREG 26 RESULT — VERDICT FALSIFIES_VOLVELLE_ADVANTAGE in both transliterations.
Best DEV config: s .45, kappa .99, b .19, V 4, rho .33, N 67, r 0.00018 (D_dev 10972.7). Top-5 all kappa high or r tiny.
Locked TEST: ZL volvelle D 11190 (repeat 115.0, vocab 11075) vs self-citation 268.5 (repeat 44.6, vocab 224.0), margin -74.2 SD. TTLI volvelle 11199 (repeat 111.1) vs self-citation 303.2 (repeat 31.9), margin -68.3 SD.
Vocabulary (ZL): attested share VMS .857 / volvelle .355 / self-cite .799; type-token VMS .358 / volvelle .268 / self-cite .410.
Reading: wear/replacement raises richness (.069 -> .268) but out-of-step rings build mostly non-VMS words (attested .355). The search therefore locks rings together (kappa .99) and almost never replaces them, which caps richness again. A ring-composition generator cannot get VMS richness AND VMS word forms at once without within-word coupling rules; the certified within-word grammar (order-2 glyph context, q->k/t, i-run endings) is exactly that missing coupling. It also loses on the repeat profile alone (115 vs 45; 111 vs 32).
Limits: ring contents are parts of random training words, not designed rings; Cham's rings would be built to give legal combinations.
P(60d) 'independent-ring volvelles with random part contents cannot reproduce VMS vocabulary + form + repeat profile': 80%.
Files: scripts/s41_volvelle_wear.py, scripts/s41_prefix.py; s41_out/results_volvelle_wear.json; s41_out/stages/*.pkl

## 2026-10-16 — PREREG 27: two interleaved streams vs one stream (written BEFORE running)
Review notes (checked first): Feb-2026 'impossibility triple' — VMS word-length autocorrelation 0.160 with low repetition and MATTR-25 0.919; copy-from-ledger generators raise autocorrelation only by destroying diversity. Word-word MI is low (0.45 bits) while within-word slot dependence is high. Oct MRC: direct lag-2 copy rule unresolved (0.24 SD); same-line shuffled donor predicts as well as the true lag-2 donor. So the two-stream idea must explain length autocorrelation without killing diversity, and must not rest on a literal lag-2 copy rule.
STAGE 1 (diagnostic, VMS ZL paragraph lines vs Vulgate pseudo-lines): within-line similarity profile using exact permutation expectations: (i) normalised edit similarity ratio obs/exp at lags 1-4; (ii) word-length excess covariance at lags 1-3. ZIG-ZAG present iff lag2 > lag1 AND lag2 > lag3 with both differences' 95% line-bootstrap CIs excluding 0. Two-stream prediction: zig-zag in VMS, absent in Vulgate. FALSIFIER for the motivation: no zig-zag in VMS on either measure.
STAGE 2 (generators; same FIT/DEV/TEST pages as PREREG 25, TEST never used to tune these families — disclosed reuse):
 STRONG generator, both arms: real-vocabulary words; mutation = replace with a real training word at edit distance 1 (grammar-preserving); page cache: with prob p draw from this page's earlier words, else global unigram; copy prob c from own-stream window w; lag-1 same-stream copy rejected with prob a.
 ONE-STREAM: K=1 (params c, w, m, a, p). TWO/THREE-STREAM: K in {2,3}, word t uses stream (t mod K) with prob q else a random stream; each stream copies from its own history (params K, q, c, w, m, a, p).
 Score D over 35 cells (24 repeat + 4 vocabulary + 4 edit-similarity + 3 length-covariance), each standardised by line-bootstrap SE on target. 150 random configs per arm FIT->DEV, top-5 rescored, best to TEST (ZL and TTLI), paired bootstrap 200.
CONFIRMS two-stream: multi-stream beats one-stream on TEST by > 2 SD in BOTH ZL and TTLI.
FALSIFIES: otherwise. Multi-stream has 2 extra params (bias toward it).
PREREG 27 STAGE 1 RESULT (within-line profile; exact permutation expectations; 95% line-bootstrap CIs):
 Edit-similarity ratio, lags 1-4: VMS 1.024, 1.063, 1.024, 1.008 -> ZIG-ZAG PRESENT (lag2-lag1 +0.039 [0.020, 0.054]; lag2-lag3 +0.039 [0.020, 0.055]).
   Vulgate 0.925, 1.006, 1.011, 1.029 (no zig-zag; neighbours strongly dissimilar). VMS token shuffle ~1.00 flat.
 Word-length excess covariance, lags 1-3: VMS 0.083, 0.001, -0.025 -> NO zig-zag: length similarity is a LAG-1 effect (lag2-lag1 -0.083 [-0.101, -0.067]). Vulgate -0.113, 0.035, 0.029.
 Reading: in form, the most similar word is two places back; in length, it is the immediate neighbour. Two different processes, or one process whose streams share length but not spelling. Pure two-stream predicts a length zig-zag too -> partly contradicted.
PREREG 27 STAGE 2 RESULT — VERDICT FALSIFIES_TWO_STREAM (one stream fits slightly better; neither margin resolved).
Best DEV: ONE c .59, w 5, m .70 (ED1 real-word mutation), a .11, p .13 (D 1874.6); MULTI K 2, q .59, c .25, w 8, m .65, a .54, p .12 (D 2020.8).
TEST ZL: ONE D 1103.9 (repeat 124.8, vocab 935.6, sim 18.8, len 24.8) vs MULTI 1181.9 (166.7, 926.8, 47.2, 41.2); margin -1.46 SD. TTLI: ONE 990.7 vs MULTI 1108.0; margin -1.88 SD.
Similarity profile ZL target 1.012 1.052 1.038 1.026; ONE 1.038 1.085 1.051 1.049 (reproduces the lag-2 peak WITHOUT streams); MULTI 0.958 1.046 0.991 1.025 (overshoots the zig-zag).
Length lag-1 covariance: target 0.068 (ZL) / 0.064 (TTLI); ONE 0.024 / 0.011; MULTI -0.014 / -0.011 -> NEITHER reproduces neighbour length pairing.
Vocabulary misfit dominates D: generators only emit attested words (attested share 1.00 vs VMS held-out 0.857) -> they cannot make the ~14% new-but-grammatical words.
Reading: the spelling zig-zag is produced by a single recency copier; two streams are not needed. Two things remain unmodelled by any generator: (1) neighbours pair by LENGTH without sharing spelling; (2) new word forms that obey the grammar.
P(60d): 'two interleaved streams not needed' 75%; 'lag-1 length pairing is a distinct unexplained constraint' 60%.
Files: scripts/s42_two_streams.py; s42_out/results_two_streams.json; s42_out/stages/*.pkl

## 2026-10-16 — PREREG 28: what produces (A) neighbour length pairing and (B) new grammatical words? (written BEFORE running)
Data: VMS ZL paragraph lines (all pages); Vulgate pseudo-lines (VMS line lengths) as reference. Lengths in atoms (VMS) / letters (Latin).
(A) Length pairing. OLS on adjacent within-line pairs: len(w[i+1]) ~ len(w[i]) + controls, line-cluster bootstrap (200) for the len(w[i]) coefficient beta.
 M0 no controls. M1 + position fixed effects (line length x position of w[i+1]). M2 + M1 + final atom of w[i] and initial atom of w[i+1] fixed effects (junction). M3 + M2 + page fixed effects.
 Rule: a control EXPLAINS the pairing if beta drops below 2 bootstrap SD (|beta|/SD < 2) once it is added; if beta stays >= 2 SD through M3, pairing is a direct length-to-length dependence not reducible to position, junction or page. Same models on Vulgate for reference.
(B) New words. Split pages as PREREG 25 (FIT+DEV = known, TEST = new). For TEST types unseen in known vocabulary (token-weighted and type-weighted), classify each: E1 = within one atom-edit of a known word; RC = splits into a known-word prefix + known-word suffix (both >= 1 atom, at any atom boundary) but not E1; NONE = neither. Null: the same classification applied to random reshuffles of the atoms within each novel word (same atoms, same length).
 Rule: a route is a GENERATIVE MECHANISM CANDIDATE if its share among real novel words exceeds the null share by > 2 SD (200 reshuffles). Report Vulgate (first 30k tokens known vs next 10k new) for reference.
PREREG 28 RESULTS
(A) Length pairing (OLS beta for len(w[i]) -> len(w[i+1]), line-cluster bootstrap):
 VMS M0 0.153 (z 24.4); +position 0.148 (24.1); +junction 0.082 (13.7); +page 0.022 (SD 0.006, z 3.6) -> survives M3 by the rule, but ~86% of it is junction + page. Position explains none.
 Vulgate: -0.127, -0.127, -0.068, -0.079 (short/long alternation, the opposite sign).
 Caveat: no line fixed effects; page FE may absorb line-level effects.
(B) New words in held-out pages (VMS 1,071 novel types = 14.3% of tokens; known = FIT+DEV):
 VMS: one-atom-edit of a known word 84.4% of novel types (atom-shuffle null 19.3%, z +78); recombination only 12.4% (null 9.0%, z +3.9); neither 3.2% (null 71.7%).
 Vulgate (letters; 30k known / 10k new): one-edit 32.9% (null 3.3%); recombination 51.8% (null 4.0%); neither 15.2%.
 Reading: VMS new words are overwhelmingly single-glyph alterations of existing words, not recombinations of parts; Latin new words are mostly stem+ending recombinations. This favours a copy-and-alter route over a slot/table/volvelle route for producing new forms.
 CAUTION: earlier record says VMS minimal-pair density sits inside the ReM (Middle German) band; Latin may be an unrepresentative reference. Needs a German (ReM) reference before reading this as VMS-specific.
P(60d): 'VMS new forms are mostly one-edit alterations of existing words (descriptive)' 85%; 'this distinguishes VMS from natural language' 40% pending ReM.
Files: scripts/s43_length_and_novelty.py; s43_out/results_length_novelty.json; s43_out/stages/*.pkl

## 2026-10-16 — PREREG 29: Ed's multi-arm (equatorium-style) coupled device (written BEFORE running)
Motivation: PREREG 28(B) says new forms are one-glyph alterations. A multi-arm device produces exactly that if ONE dial moves per reading while the others hold, and several sub-disks read in turn give stream-like recurrence (sub-disk k recurs every A readings).
MULTI-ARM device (6 params): A sub-disks (1-4), each with 4 rings of N cells (12-120); columns = real training words split into 4 parts (home alignment emits real words); odd columns blank with prob b. Readings cycle through the sub-disks in order with prob q, else a random sub-disk. After a reading, with prob z all rings of that sub-disk advance to the next column together (new real word); otherwise exactly ONE ring (chosen at random) moves +1 (one-part change). Each word, with prob r one ring of the current sub-disk is replaced by a fresh ring.
Rival: the PREREG 27 ONE-stream copier (its best config, same split; not re-tuned).
Budget: 150 random device configs FIT->DEV (2 samples), top 5 rescored (5), best to TEST (ZL and TTLI), paired bootstrap 200. Score = the same 35-cell D as PREREG 27.
CONFIRMS device advantage: device beats the copier on TEST by > 2 SD in both ZL and TTLI. FALSIFIES otherwise. Also reported: attested share and one-edit share of the device's novel words (VMS: novel 14.3% of tokens, 84% one-edit).
Disclosure: TEST pages reused (fourth time); the device family was never tuned on them.
PREREG 29 RESULT — VERDICT FALSIFIES_DEVICE_ADVANTAGE (ZL +1.44 SD in device's favour, below 2; TTLI -7.0 SD against).
Best DEV: A 4 sub-disks, N 119, b .19, q .48, z .77, r .0003 (D_dev 1510 vs copier 1875 -> device better on DEV).
TEST ZL: device 1091 (repeat 77.8, vocab 970, sim 9.0, len 34.8) vs copier 1236 (repeat 135.8, vocab 1052, sim 14.7, len 33.0). TTLI: device 1625 (repeat 67.8, vocab 1503) vs copier 904 (repeat 91.2, vocab 774).
Device beats the copier on the REPEAT profile in both transliterations (77.8 vs 135.8; 67.8 vs 91.2); it loses on vocabulary (richness .21 vs VMS .36; attested .78/.74 vs .86/.87; novel tokens 31-34% vs VMS 14%).
Device novel words are 83-85% one-glyph edits (VMS 84%) -> matches HOW new words are made, overproduces HOW MANY.
Neither model reproduces neighbour length pairing (device ~0.00; copier .02; VMS .068).
METHOD NOTE: the copier's TEST D differs between runs of the same config (ZL 1103.9 in PREREG 27 vs 1235.7 here) -> generator sampling noise is large and NOT included in the paired bootstrap SD (which resamples held-out lines only). Margins near 2 SD are not trustworthy; future runs need many more generated samples or a generator-variance term.
P(60d): 'multi-sub-disk one-dial device fits the repeat profile better than a recency copier' 50%; 'overall advantage' 20%.
Files: scripts/s44_multiarm_device.py; s44_out/results_multiarm_device.json; s44_out/stages/*.pkl

## 2026-10-16 — PREREG 30: can a realistic ring device's REACHABLE vocabulary contain the Voynich vocabulary? (written BEFORE running)
Ed's reframing: the VMS vocabulary need only be a SUBSET of what the device can produce. Test coverage, with specificity controls so a device that can write anything does not win by default.
Prior record: CRIB-SHEET/FORM work found ~32 learned chunks license unseen types (49 pp over matched null); this test asks the physical-device version of that question.
DEVICE: R rings in fixed order; word = one cell from each ring, concatenated (cells may be blank). Rings learned from training (FIT+DEV) words by greedy budgeted cover: words in descending frequency; each uncovered word is split into R contiguous atom chunks choosing the split needing fewest new cells; cells added while each ring is under its budget C.
 FREE variant: rings turn independently (reachable set = all combinations).
 COUPLED variant (notched rings, equatorium-style linkage): a cell pair on adjacent rings is reachable only if that pair occurs in some training parse.
Grid: R in {3,4,5,6}; C in {12,24,36,48}. "Realistic" = R <= 6 and C <= 36.
Evaluation on held-out TEST pages: token coverage; novel-type coverage (types absent from training); DECOY coverage = the same novel types with atoms shuffled within each word (50 shuffles); reachable-set size (FREE exact product; COUPLED by path counting).
Natural-language control: identical procedure on Latin (Vulgate; letters; known 30k tokens, new next 10k).
SOLVES THE WORD PROBLEM (for VMS): some realistic config has TEST token coverage >= 95%, novel-type coverage >= 80%, and decoy coverage <= 0.5 x novel-type coverage.
FALSIFIES: no realistic config meets all three.
Interpretation guard: if Latin meets the same criteria with a comparable device, then coverage shows only that VMS words are morphologically regular, not that a device made them.
PREREG 30 RESULT — VERDICT FALSIFIES_REALISTIC_DEVICE_COVERAGE for the greedily-learned free/notched ring device (0 configs meet all criteria).
Best realistic VMS (6 rings x 36 cells): FREE test tokens 81.2%, novel types 36.9%, decoys 5.3%; COUPLED 80.3% / 31.9% / 3.6%. 5x48: 82.4% / 38.3% / 4.6%. Budget stops binding by 48 cells (learner limit, MAXCHUNK 4).
Latin, same device: 6x36 FREE 33.9% tokens, 1.3% novel. -> VMS words are far more device-friendly than Latin words (ring devices cover ~2.4x the held-out tokens), and specific (decoys ~1/7 of novel coverage).
Caveat: greedy frequency-order learner, not an optimised design; coverage is a floor.

## 2026-10-16 — PREREG 31: Ed's 3-dial Stolfi device (crust / mantle / core), read across a diameter (written BEFORE running)
Idea: Stolfi's NormalWord is crust-mantle-core-mantle-crust (record: covers 95.7% of ZL test tokens). Three concentric dials read across a diameter give exactly that symmetric layering: the crust dial supplies a (left, right) crust pair, the mantle dial a (left, right) mantle pair, the core disk one core.
Operational layers (atoms): CORE = gallows k t p f and benched ckh cth cph cfh; MANTLE = ch, sh, e; CRUST = all other atoms. Parse = leading crust run, mantle run, core run (may be empty), mantle run, trailing crust run; words violating the onion order are unparseable (counted uncovered).
DIAL device: each dial holds N cells; crust/mantle cells are (left,right) pairs; core cells single chunks; contents = the N most frequent components in training (FIT+DEV). Word reachable iff its parse exists and all three components are on their dials.
RIVAL with identical parse and cell count: FIVE independent slots (cL, mL, core, mR, cR), each with the top-N chunks (unpaired) -> isolates the effect of the diameter pairing.
N in {12, 24, 36, 48, 72, 96}; realistic = N <= 36 per dial.
Same criteria as PREREG 30: SOLVES if a realistic config has TEST token coverage >= 95%, novel-type coverage >= 80%, decoy (within-word atom shuffle) coverage <= 0.5 x novel. Report onion-parse rate of TEST tokens (ceiling).
PREREG 31 AMENDMENT (written before any run; nothing executed yet):
(a) Stolfi treats circle letters o, a, y as able to sit in any layer. PRIMARY parse stays strict as above; pre-specified SECONDARY = CIRCLES variant where o/a/y may belong to any layer. Both reported; verdict computed on each, and SOLVES requires the criteria in either one, flagged by variant.
(b) Cell budget: DIAL uses 3N cells. Rivals: FIVE_N (N per slot, 5N cells, generous) and FIVE_MATCHED (round(3N/5) per slot, 3N cells). DIAL "beats" a rival if decoy-adjusted coverage (novel - decoy) is higher at the same N.
(c) Canonical parse for learning dial contents: circles take the layer of the preceding non-circle atom (crust at word start); unresolved mantle with empty core goes left. Coverage at test time accepts ANY valid parse.
PREREG 31 RESULT — VERDICT FALSIFIES_STOLFI_DIAL_COVERAGE (0 solving configs, either parse variant). scripts/s46_stolfi_dials.py, s46b_checks.py, s46c_naive.py; s46_out/results_stolfi_dials.json.
Grammar ceiling: onion parse covers 94.8% TEST tokens (STRICT) / 97.8% (CIRCLES), consistent with Stolfi's ~95%. BUT it also parses 52.6% / 68.3% of atom-shuffled decoys (novel types 75.7% / 85.5%) -> the onion order alone is permissive (novel/decoy ratio 1.44 / 1.25: weak specificity).
3-dial device, N=36 per dial (108 cells): TEST tokens 58.3%, novel types 8.3%, decoys 1.3% (STRICT); CIRCLES 66.5% / 10.4% / 2.6%. Even N=96: 75.6% / 17.4%.
Rivals at N=36: FIVE_MATCHED (108 cells) 83.4% / 28.9% / 4.5%; FIVE_N 87.5% / 39.5% / 5.0%. DIAL loses to both at every N (novel-minus-decoy deficit -0.02 to -0.33). Exact counts, no resampling SD; differences are many times binomial SE (n_novel ~ 2000 types).
Positive control (text generated by a 36-cell 3-dial device): learner recovers 96.7% tokens, 87.1% novel types, 3.2% decoys -> the test can detect a dial device when one exists.
Why the dials lose: L/R crust are dependent (MI 0.752 bits vs 0.304 shuffled, z 105) and L/R mantle too (0.089 vs 0.006, z 128), so pairing is real, but VMS uses far more L/R combinations than 36 paired cells allow; pairing is too rigid.
Rule-19 naive baseline: first_2/last_2 excess MI 0.333 bits vs Stolfi crust excess 0.448: ratio 1.35 -> the metric does not resolve whether Stolfi layers carry more L/R coupling than a naive first/last partition.
Descriptive: word beginnings and endings are coupled (both partitions). Analytical claim ("dial pairing captures it") is not supported: a paired-dial device covers less than independent slots.
P(survives 60d): verdict 85%; "onion order is permissive vs shuffles" 75% (depends on my layer assignment, not Stolfi's exact paradigm).

## 2026-10-16 — PREREG 32: salvage of the Stolfi-dial idea (written BEFORE running)
Already-computed (descriptive, no new run): at ~108 cells, Stolfi-layer independent slots (FIVE_MATCHED) 83.4% tokens / 28.9% novel / 4.5% decoy vs PREREG 30 learned rings 3x36 70.6% / 15.1% / 1.5%; at 180 cells FIVE_N 87.5 / 39.5 / 5.0 vs rings 4x48 82.1 / 37.1 / 4.4.
S1 (layout): Stolfi 5-slot device vs NAIVE 5-slot device (fixed split first1, second, middle, second-last, last; deterministic) at N in {24,36,48}. Stolfi advantage claimed only if Stolfi beats naive on test-token coverage by >= 5 points at every N with decoy rate no worse than 1.5x naive's. Otherwise "layout advantage not resolved".
S2 (mostly-aligned dial = pairs as frequencies, not reachability): P(l,r) = a*A36(l,r) + (1-a)*P(l)P(r), A36 = top-36 training pairs. Train FIT, fit a (and joint weight b) on DEV, score TEST cross-entropy. captured = (H_indep - H_mix)/(H_indep - H_joint). CONFIRM alignment-explains-coupling if Stolfi crust captured >= 0.50 AND Latin naive (first2,last2) captured is < VMS value - 0.10. FALSIFY if VMS captured < 0.25 or Latin >= VMS - 0.10 (non-diagnostic). Rule 19: VMS naive (first2,last2) reported alongside.
S3 (device + copying slips): FIVE_N N=36 Stolfi STRICT and naive device; novel types covered exactly OR one atom-edit (sub/ins/del) from a covered word; decoys (3 atom shuffles per novel) the same. SOLVES if a device reaches novel >= 80% with decoy <= 0.5 x novel. FALSIFY otherwise.
PREREG 32 RESULT (scripts/s47_salvage.py, s47b_bounds.py; s47_out/results_salvage.json)
S1: NOT MET as pre-registered — naive 5-slot beats Stolfi 5-slot on tokens (N36: 89.9% vs 87.5%). POST HOC (flagged): naive is non-specific (decoys 36.4% vs Stolfi 5.1%; novel/decoy 1.5 vs 7.7). Specificity comes mostly from the onion order rejecting shuffles (known property of Stolfi's grammar).
S2: crust alignment captured 0.415 -> between thresholds, UNRESOLVED (neither confirm nor falsify). Latin naive 0.225, VMS naive 0.335 (Stolfi/naive 1.24 -> metric does not resolve Stolfi vs naive). Mantle 0.944 is trivial: only 85 distinct mantle pairs, top-36 hold 99.5% of mass.
S3: criterion MET as pre-registered — Stolfi 5x36 device + one atom slip: novel 84.7%, decoys 30.6% (<= 0.5x); tokens 97.7%. Naive device + slip fails (88.7% / 75.7%).
BOUND (rule 17, post hoc): "known training word + one slip" gives tokens 97.9%, novel 84.4%, decoys 18.5% — same coverage, BETTER specificity. So S3 shows a realistic device+slip superset contains ~98% of held-out tokens (Ed's superset question: yes), but it does NOT discriminate device from copying. The S3 criterion was too weak (no copy comparator); recorded as such, not as device evidence.
P(survives 60d): S3 existence result 75%; "S3 is non-diagnostic vs copying" 85%; S1 specificity contrast 70%.

## 2026-10-16 — PREREG 33: device vs copying at the level of local word-to-word relations (written BEFORE running)
Question: when nearby words in a line are related, is the relation a DIAL TURN (one whole Stolfi segment swapped, often >= 2 glyphs) or a COPY SLIP (one glyph edited)?
Per line, pairs at lag d = 1..4: E1 = atom Levenshtein exactly 1; SW = differ in exactly one Stolfi segment (STRICT canonical parse) with edit distance >= 2. Each as observed / exact within-line permutation expectation.
Discriminator D1 = SW(1) / E1(1) (swap-likeness at lag 1). Secondary D2 = SW(1) - SW(2) (lag-1 primacy expected of a sequential dial).
Positive controls (must separate or the test is VOID): DEVICE family = 5 Stolfi dials x 36 cells (top-36 training components, random cell order), each dial turned with prob t in {.2,.4,.6} by up to s in {1,3} cells, slip prob m in {0,.1}; COPY family = copy a word from the last w in {4,10} tokens with prob c in {.3,.5,.7}, slip prob m in {.2,.5}, else draw from training unigram. 10k tokens each, VMS line lengths.
VOID if the device-family D1 range and copy-family D1 range overlap. Otherwise VMS (all pages, ZL; replicated TTLI) with line-bootstrap 95% CI: DEVICE-LIKE if the CI lies inside or beyond the device range and clear of the copy range; COPY-LIKE vice versa; UNRESOLVED if it touches both or falls between.
Rule 19 baseline: same with the naive 5-slot split (first, second, middle, second-last, last) in place of Stolfi segments.
PREREG 33 RESULT — VOID on the primary metric (scripts/s48_turn_vs_slip.py; s48_out/results_turn_vs_slip.json). D1 control ranges overlap (Stolfi: device 0.825-1.099, copy 0.727-1.059; naive likewise). VMS D1 0.937 [0.891, 0.981] ZL, 0.947 TTLI — uninformative.
Secondary D2 (pre-specified as secondary; decision rule NOT pre-specified for it -> flagged): device 0.306-3.541, copy -0.116-0.070 (separated). VMS 0.072 [0.034, 0.119] ZL; 0.088 [0.031, 0.140] TTLI: touches the copy range, clear of the device range.
POST HOC (flagged): one-edit enrichment lag1 minus lag2: device 0.21-3.21, copy -0.05-0.08, VMS 0.018 (ZL) / 0.011 (TTLI). VMS E1 by lag 1.13, 1.11, 1.02, 1.04 — flat over lags 1-2, then drops (copy window ~2; consistent with SGT12 lag-2).
Free-spin bound (s48b_freespin.py): spinning all dials far each word kills local relations entirely (E1 ~ 0 at all lags) — unlike VMS's 1.13. A sequential device either gives lag-1 primacy or no relations; VMS has neither.
Reading (analytical claim — needs external test): local relations in VMS look like copying from the last ~2 words with one-glyph slips, not like turning dials. Limit: device family is single-reader sequential; the multi-arm device (PREREG 29) can produce lag-2 structure and was not in this control family.
P(survives 60d): VOID 95%; D2/lag-flatness contrast "copy-like vs sequential dial" 65% (post hoc, single session).

## 2026-10-17 — PREREG 34: does npcompl33t's per-bifolium ordering stability survive TEXT resampling? (written BEFORE running)
Source: eo777.github.io/voynich-clusters/chronology-stability/individuals.html (repo cloned to /home/claude/ext/voynich-clusters, 17 Oct).
His ensemble (ENSEMBLE_DESIGN.md) keeps the MDS coordinates FIXED and drops 10% of points; it "neither repeats distance estimation/MDS". So the
page's near-certain pairwise orders measure curve-fit robustness to point omission, not uncertainty from the finite text on each sheet.
Test (pool50, lambda 0.01, his code unchanged: core.distance_from_counts, COMMON.embed, his 8 declared starts rebuilt from group centroids in each
new geometry, grow/orient/refine/project, "best" = lowest objective among final 1a-before-1b): B = 100 replicates; each replicate resamples every
pool's word and line transition rows multinomially with row totals kept, then recomputes JS -> MDS -> curve. No point masks (isolates text noise).
This multinomial resampling ignores clustering of transitions within words/lines, so it UNDERSTATES text noise: a lower bound.
Reproduction gate (must pass first): the unresampled run must reproduce his full-data best-curve pool order (Spearman >= 0.95 vs the median position
of each pool in his 'best' scope); otherwise VOID.
Metric: share of the 1225 pool pairs that are decisive (one direction in >= 95% of replicates; ties count half) — ours (TEXT) vs his best-scope pool50 l0.01 (POINT).
Also median per-pool 5–95% position span in each.
FALSIFIES "the displayed stability reflects the text": TEXT decisive share < 0.8 x POINT share. CONFIRMS: TEXT >= 0.95 x POINT. Between: partial.
Focal check (his own finding I:67-68 after 2a pools; f86v4 needs paragraph207, not run): report TEXT direction fractions for I:67-68 vs each 2a pool.
PREREG 34 RESULT (scripts/s49_text_resampled_order.py, s49b_compare.py; s49_out/results_text_resampled.json)
Gate PASSED: unresampled rerun of his code reproduces his best-scope pool order (Spearman 0.998).
Multinomial text resampling (100 reps, all succeeded): decisive pairs 91.2% vs his best-scope 92.3% -> ratio 0.988 -> CONFIRMS as registered
(his displayed per-bifolium stability is not inflated by ignoring transition-count noise, at this lower-bound noise level).
Median 5-95% position span 0.094 (text) vs 0.076 (his best scope); reversals of his majority direction 0.2% of pairs. Rep vs full-data Spearman median 0.996.
Group-level: 3a before 3b in 77% of reps (his "unsettled" verdict holds; text noise favours 3a-first); 4a before 4c 2%.
His focal: I:67-68 after 2a pools in 94-100% of text reps.
Headline AWAITING BOUNDING: multinomial ignores within-line clustering. PREREG 34b below.

## 2026-10-17 — PREREG 34b: line-block bootstrap bound (written BEFORE running)
Rebuild per-line ZL counts with his own SourceLattice/glyph_units code (gate: summed lines must equal his pool counts EXACTLY, else VOID).
Each replicate resamples whole transcription lines with replacement within each bifolium (paragraph lines, same count), then the identical s49 pipeline. B = 100.
Same criteria vs his best scope: ratio < 0.8 FALSIFIES; >= 0.95 CONFIRMS; between PARTIAL. Also report vs the multinomial result.
PREREG 34b RESULT (scripts/s50_line_bootstrap_order.py; s50_out/results_text_resampled.json). Gate PASSED: per-line ZL counts rebuilt with his toolkit sum EXACTLY to his pool counts (4,130 lines).
Line-block bootstrap, 100 reps: decisive pairs 90.0% vs his best scope 92.3% -> ratio 0.974 -> CONFIRMS as registered. Median span 0.101 vs 0.076; reversals of his majority 0.4%.
3a before 3b in 62% of reps (unsettled, as he says); 4a before 4c 1%. I:67-68 after the 2a pools 88-98%.
Reading (descriptive): the per-bifolium ORDER on his individuals page is robust to the finiteness of the text, not only to dropping points. This says nothing about whether the order is chronological (he says so himself); our PREREG 12 found the gradient partly coincides with the Currier A->B axis.
P(survives 60d): "his pool50 ordering is robust to text resampling" 85% (single cohort/penalty; paragraph207 untested).
Least stable under line bootstrap: J:69-70 (5a, 137 words vs median 418): 5-95% span 0.298 vs 0.072 on his page (4x understated); D:26-31 0.181 vs 0.122; H:57-66 0.170 vs 0.124. Small-text bifolia are where his page overstates certainty.

## 2026-10-17 — PREREG 35: is npcompl33t's order useful beyond glyph style? Does it predict shared WORDS? (written BEFORE running)
Order = his published pool50 positions (ZL, lambda 0.01, best-scope median, individuals data.json). Built only from glyph transitions.
Held-out feature: word vocabulary. Per bifolium, paragraph tokens (ZL via his SourceLattice, readable only); similarity = cosine of
type-frequency vectors over types with corpus frequency >= 3. Replication: TTLI tokens (our loader, P-lines) mapped page -> bifolium.
Strata: section x LFD scribe (majority of member pages): 44 of 50 bifolia in strata of size >= 2 (H|1 = 24, S|3 = 7, H|2 = 5, B|2 = 5, P|1 = 3).
Primary statistic: over same-stratum pairs, Spearman(|position difference|, similarity residualised on log size_i + log size_j).
Null: permute positions within strata, 10,000 draws. CONFIRM "order predicts vocabulary beyond section and scribe": z <= -2 and permutation p < 0.05
in ZL AND TTLI. FALSIFY: |z| < 1 in ZL. Otherwise UNRESOLVED. Effect size and null SD stated together.
Secondary (the decoding-relevant one): residualise similarity ALSO on his pairwise glyph-transition distance (combined JS). If the 1-D order still
predicts vocabulary (z <= -2), it carries word-level information beyond glyph-transition style. If not, the order's vocabulary signal is just
glyph style restated.
Prior: P(primary CONFIRM) 0.7 (PREREG 12 found a gradient in H|1); P(secondary CONFIRM) 0.3.
PREREG 35 RESULT (scripts/s51_order_vs_vocabulary.py; s51_out/results_order_vocabulary.json)
PRIMARY CONFIRMS: same-stratum pairs 320; ZL rho -0.688 vs null -0.072 (SD 0.067), z -9.13; TTLI -0.676, z -8.96. Bifolia closer in his order share more words, within section x scribe.
SECONDARY CONFIRMS (letter of prereg): after removing his glyph-transition distance (linear), ZL rho -0.159 vs null 0.006 (SD 0.064), z -2.57, p 0.009; TTLI -0.143, z -2.40, p 0.011.
Effect/null-SD ratio ~2.5: small effect. AWAITING BOUNDING -> PREREG 35b.
## PREREG 35b bound (written before running): secondary re-run with (a) flexible transition control: word-JS, line-JS, combined, combined^2, rank(combined);
(b) (a) plus rarefied similarity (each bifolium subsampled to 120 tokens, mean of 50 draws). Secondary survives bounding only if z <= -2 in ZL and TTLI under (b).
PREREG 35b RESULT (scripts/s51b_bound.py; s51_out/results_bound.json): verdict by the registered criterion SURVIVES BOUNDING:
(b) flexible transition control + rarefied similarity: ZL rho -0.219 vs null 0.029 (SD 0.067), z -3.72; TTLI -0.214, z -3.73.
BUT (a) flexible control on raw similarity: ZL rho -0.087 vs 0.027 (SD 0.065), z -1.74; TTLI z -1.78 -> below 2: the metric does not resolve this specification.
Reading: the order's large vocabulary association (rho ~ -0.69) is mostly glyph-transition style restated (falls to ~ -0.09 to -0.22 once transitions are controlled).
A small residual word-level signal is present and specification-sensitive. Analytical claim "the order carries word information beyond glyph style": needs external test before relying on it.
P(survives 60d): primary 90%; secondary 50%.

## 2026-10-17 — PREREG 36: respelling drift WITHIN one scribe and section along npcompl33t's order (written BEFORE running)
Why: earlier respelling tests (PREREG 19-21) compared his stages across scribe/section changes (confounded) and exact-form context tests were underpowered.
Here: herbal scribe 1 only (H|1, 24 bifolia; section and scribe fixed), his published pool50 positions (ZL, lambda .01, best-scope median).
Tokens: ZL paragraph tokens (his SourceLattice), replication TTLI; words as atom strings (ch, sh, ckh, cth, cph, cfh = one atom).
Types with >= 8 occurrences in H|1. trend(w) = Spearman(position, relative frequency of w per bifolium) over the 24 bifolia.
Opposed pair = one declining (trend <= -0.4) and one rising (trend >= +0.4). Statistic D = opposed-pair rate among one-edit (atom Levenshtein 1) pairs
minus opposed-pair rate among far pairs (distance >= 3). Respelling (w gradually replaced by a one-glyph variant) predicts D > 0.
Null: permute positions among the 24 bifolia (2,000 draws), recompute everything; z = (D - null mean)/null SD.
CONTROLS BUILT AND RUN FIRST (VOID if they fail): positive = random order + 20 injected respellings (a frequent type w replaced by an existing
one-edit type v with probability rising 0 -> 1 along the random order); must give z >= 2 in >= 16/20 runs. Negative = random order, no injection:
|z| < 2 in >= 18/20.
CONFIRM respelling drift along his order: z >= 2 in ZL AND TTLI. FALSIFY: |z| < 1 in ZL. Else UNRESOLVED. Sensitivity thresholds 0.3 and 0.5 reported.
Caveat fixed in advance: a pass shows paired decline/rise of one-glyph variants along the order within one scribe; it does not show the variants mean the same thing.
PREREG 36 RESULT (scripts/s52_respelling_within_scribe.py; s52_out/results_respelling_within_scribe.json) — VERDICT UNRESOLVED
Controls VALID: positive (20 injected respellings, random order) z >= 2 in 17/20 ZL, 19/20 TTLI; negative |z| < 2 in 19/20 both.
Main (H|1, 169 types, 548 one-edit pairs): ZL D 0.0020 vs null -0.0008 (SD 0.0017), z 1.65; TTLI z 0.48 -> not confirmed, not falsified (ZL |z| >= 1).
Sensitivity: thr 0.3 z -3.09 (ZL) / -1.62 (TTLI); thr 0.5 z -12.8 (0 opposed one-edit pairs; degenerate, few counts) -> if anything one-edit pairs are LESS often opposed than far pairs.
POST HOC (flagged; s52b_cotrend.py): one-edit neighbours trend in the SAME direction far more than far pairs: ZL excess 0.026 vs null 0.001 (SD 0.003), z 8.4 (thr .4); TTLI z 8.3; thr .3 z 5.4 / 4.4.
Interpretation limit: largely CIRCULAR — the order is built from glyph transitions, and one-edit neighbours share most transitions, so co-trending along it is expected by construction. Not evidence of anything beyond the order's definition.
Reading for decoding: within scribe 1 the order does not expose word-for-word one-glyph replacements; variant families rise and fall together. P(survives 60d): "no detectable replacement signal within H|1" 70%.

## 2026-10-17 — IMPACT CHECK: his order is far sharper than the gradient PREREG 13 assumed
Within herbal scribe 1 (24 bifolia), rank SD of each bifolium across text resamplings of HIS order: median 0.85 (multinomial) / 1.07 (line bootstrap), max 2.7.
PREREG 13's power analysis assumed rank SD 3.8 (our own rarefied Fiedler gradient, 21 sheets) and concluded text "cannot resolve individual neighbours".
That conclusion applies to OUR gradient, not to his order. Consequence: the physical-anchor test was underpowered for the wrong reason and can be re-run on his order.
(Caveat: 1.07 is text noise only; it excludes his start/penalty variation. Our 3.8 came from rarefaction to equal size, which adds noise.)

## PREREG 37 (written BEFORE looking at his positions for the anchor sheets): physical anchors vs his order
Exposure note: I have seen his ID list for H|1 and one position (B:11-14 ~0.147, from the earlier span table); not the anchor ranks.
Anchors as PREREG 13 (Pelling): V vellum flaws E:35-38~E:36-37, E:36-37~B:9-16, B:9-16~B:10-15; T contact transfers A:2-7~A:3-6, A:3-6~A:4-5,
B:10-15~B:11-14, C:19-22~C:20-21. All in H|1.
Statistic: mean |rank difference| over anchor pairs, ranks of his best-scope median positions among the 24 H|1 bifolia. Null: permute ranks (100k).
Report V, T, V+T. Noise propagation: same statistic in each of the 100 line-bootstrap replicates (share with z <= -2).
Power: simulate truly adjacent pairs with rank noise SD 1.07.
CONFIRM "his order agrees with physical adjacency": V+T z <= -2 AND >= 80% of bootstrap reps z <= -2. FALSIFY: V+T |z| < 1 while power >= 80%. Else UNRESOLVED.
What a pass would mean: physically adjacent sheets sit near each other in a text order built without any physical information. It would not prove the order is writing order.
PREREG 37 RESULT (scripts/s53_anchors_his_order.py; s53_out_anchors.json) — VERDICT FALSIFIES (see R5 at top).
V: 3.00 vs 8.33 (SD 3.32) z -1.60 p 0.035; T: 9.00 vs 8.34 (SD 2.76) z +0.24; V+T: 6.43 vs 8.32 (SD 2.08) z -0.91 p 0.20. Bootstrap reps: z 5/50/95% -0.98/-0.56/-0.12, 0% reach -2.
Power with true adjacency and rank noise SD 1.07: P(z <= -2) = 1.00 (excludes his start/penalty variation and any loose link between nesting and writing order -> overstated).
P(survives 60d): 70% (anchor validity is the weak point: transfers partly disputed; only f2v->f3r judged original by Pelling, and it is the worst pair, 15 ranks).

## 2026-10-17 — PREREG 38: do random page splits flatter the novelty/coverage results? + a direction test (written BEFORE running)
Context: PREREG 28 (84% of held-out new types one atom-edit from training vocab) and PREREG 32 bound (vocab+slip covers ~98% tokens) used random page splits.
Vocabulary shifts along npcompl33t's order within scribe 1 (PREREG 35), so random splits put neighbouring stages on both sides.
Data: herbal scribe 1 (H|1, 24 bifolia), ZL (his SourceLattice tokens) and TTLI; words as atom strings. Order = his pool50 best-scope median positions.
A) SPLIT TEST. Early half (12 bifolia) vs late half (12) by his order. ORDER split: train early -> test late, and train late -> test early.
   RANDOM split: 1,000 random 12/12 partitions. Metrics on test: E1 = share of novel types within one atom edit of a training type; COV = share of test
   tokens that are known or one edit from known. Both halves rarefied to the smaller half's token count (mean of 20 draws) so size cannot drive it.
   FLATTERED if both ORDER-split E1 values fall below the 5th percentile of RANDOM E1 (both transcriptions). NOT FLATTERED if both lie above the 25th percentile.
B) DIRECTION. A = E1(early->late) - E1(late->early). Self-citation along a true writing order predicts A > 0 (new forms derive from earlier forms).
   Null: 1,000 random permutations of the 24 positions, split at median, same A. DIRECTIONAL if z >= 2 in ZL and TTLI (then reverse-direction z <= -2 would
   mean his order runs backwards). NO DIRECTION if |z| < 1 in ZL.
PREREG 38 RESULT (scripts/s54_order_split_novelty.py; s54_out/results_order_split.json)
A) SPLIT: UNRESOLVED by the registered rule (needs both directions). Train EARLY -> test LATE is below the random-split range: E1 ZL 0.712 (3rd pct; random median 0.749),
   TTLI 0.757 (0th pct; median 0.812); token coverage ZL 0.918 (1st pct; median 0.938), TTLI 0.936 (0th pct; median 0.957). Train LATE -> test EARLY is ordinary
   (E1 pct 0.32 / 0.62). So random splits flatter forward prediction by ~4-5 points of E1 and ~2 points of coverage. PREREG 28's "new words are mostly one edit from known
   words" STANDS (71-82% under order splits vs atom-shuffle null ~19%); its exact 84% is a few points optimistic.
B) DIRECTION: NO_DIRECTION by rule (ZL |z| < 1): A = E1(early->late) - E1(late->early) ZL -0.028 vs null 0.000 (SD 0.033), z -0.86; TTLI -0.060, z -1.88.
   Both lean the same way: the late half predicts the early half's new words better than the reverse — the opposite of a simple "later derives from earlier"
   reading of his order. Not resolved; do not interpret.
P(survives 60d): "random splits flatter forward novelty by a few points" 70%; direction: unresolved.

## 2026-10-17 — DEPENDENCY SCAN of prior work against today's order results (task "check everything")
- PREREG 13 (anchors): SUPERSEDED — see R5 at top; re-run on his order FALSIFIES physical-adjacency agreement (PREREG 37).
- PREREG 12 (gradient in H|1, z 2.52; rank SD 3.8): STANDS as a statement about our gradient; his order is the sharper instrument (rank SD ~1.1).
- PREREG 19-21 (respelling): STANDS; reading sharpened — within-scribe replacement absent (PREREG 36), so the positive A->B steps coincide with scribe change.
- PREREG 28 (84% one-edit novelty): STANDS qualitatively; random split flatters forward prediction by ~4-5 points (PREREG 38).
- PREREG 30-32 coverage (vocab+slip ~98% tokens): STANDS; order split lowers forward token coverage by ~2 points (0.918-0.936 vs 0.938-0.957 random median).
- PREREG 33 (turn vs slip): UNAFFECTED (within-line permutation statistics; stage mixing cancels).
- PREREG 24-29, 39-44 generator fits: scored on randomly drawn TEST pages; vocabulary cells (attested share, TTR) are mildly flattered for the same reason; repeat-profile cells are within-line and unaffected. Not re-run.
- Super Grammar SG23_REMOTE_STATE_REUSE ("recur on remote pages ... rather than purely continuous page drift"): its contrast was against drift in PHYSICAL order.
  Post-hoc check (s51c_crossquire.py): across DIFFERENT quires, same section x scribe, vocabulary similarity still follows his order: ZL rho -0.673 vs null -0.030 (SD 0.075),
  z -8.54; TTLI -0.665, z -8.44. So "drift along a writing order unrelated to the binding" is a live alternative SG23 did not test. CIRCULARITY caveat: his order is
  built from page similarity, so similar remote pages will always sit close in it; this check cannot by itself separate drift (1-D) from recurring states.
  Needed: 1-D order vs K-state partition model comparison on held-out pages. SG23 interpretation -> OPEN (statement about remote recurrence itself stands).
- SG47, SG55, SG56, SG22: no conflict found (SG56 four-state plan is compatible with stages along an order).

## 2026-10-17 ~23:05 — METHOD DECLARED BEFORE RUNNING: combined-uncertainty stage map of npcompl33t's pool50 order (descriptive, no hypothesis)
Each of 100 replicates b: whole-line bootstrap of text within bifolia (s50) AND his own Bernoulli(0.9) point mask b (from his ensemble design) AND all 8 of his starts,
lambda 0.01 and 0.05 (1,600 fits). Positions recorded for retained bifolia only (his convention). Two scopes: BEST (lowest objective per replicate x lambda among 1a-before-1b)
and ALL (every 1a-before-1b fit, equal weight). Pair fraction = share of jointly retained fits with i before j (ties half).
Stage map: order by mean pairwise 'after' share (his consensus rule); a boundary between ranks k and k+1 is SETTLED if every pair across it (i <= k < j) has
i-before-j >= 0.95. Blocks between settled boundaries = stages whose internal order is not settled. Reported for both scopes and compared with his page.

## 2026-10-17 ~23:20 — PHYSICAL ANCHOR SOURCES (looked up via Exa; primary pages read in highlights)
Pelling, "Voynich Codicology" (ciphermysteries.com/the-voynich-manuscript/voynich-codicology, 2008, list of "likely ORIGINAL page adjacencies"):
 (1) f2v->f3r faint ink/paint contact transfers "appear to be original" (Curse pp.65-67) -> A:2-7 ~ A:3-6.
 (2) vellum flaws (Curse pp.53-56): single quire f35-f36-f9-f10-(centre)-f15-f16-f37-f38, "possibly with the f28-f29 bifolio wrapped around" -> E:35-38~E:36-37~B:9-16~B:10-15 (+ D:28-29 outer, "possibly").
 (3) Q13 original order f76-f77-f79-f84-f78-(centre)-f81-f75-f80-f82-f83 (Curse pp.62-65); REVISED by Pelling 2021 (ciphermysteries 2021/07/24): Q13 = two gatherings,
     Q13B = f84-f75 wrapped around centre f78-f81 ("rock solid"); Q13A order disputed. Davis (manuscriptroadtrip 2025-01-19) independently: f78v/f81r was the centre bifolium.
 (4) Q9/Q10 reconstruction; zodiac starts at Pisces; f116v last page (not bifolium-pair anchors in the pool50 set).
 Pelling 2009/03/27 ("Voynich herbal page order"): f2v->f3r light transfer = Pro (original); blue/red transfers (f3v/f4r, f5v/f6r, f19v/f20r, f15r->f14v) are DISPUTED (later paint).
 -> PREREG 13/37's T set mixed one original (f2v->f3r) with disputed later-paint transfers in the PRESENT binding. Confirms R6(b).
Davis 2025: upper-margin water stain in the first seven quires predates the misordering; stain shape changes abruptly between current neighbours (f32v vs f33r).
 -> A stain-shape continuity measure across all herbal leaves could supply MANY original-adjacency anchors (image work; not doable here without image access).
CLEAN ORIGINAL-STATE SET available now: E:35-38~E:36-37, E:36-37~B:9-16, B:9-16~B:10-15, A:2-7~A:3-6 (herbal); M:75-84~M:78-81 (bio; Q13B).
 BLINDNESS: the four herbal pairs' ranks in his order were already seen (PREREG 37): vellum 3 pairs mean |rank diff| 3.0 (p 0.035), f2v-f3r 15 ranks apart.
 So no confirmatory test is possible with literature anchors; the bio pair sits where his positions tie at the curve end (cold check). Text-vs-physical remains UNRESOLVED; the route is stain-shape anchors from images.
STAGE MAP RESULT (scripts/s55_combined_stage_map.py, s55b_stage_map.py; s55_out/stage_map.json, stage_map_combined_best.csv; 200 replicate sets, 1,600 fits, all completed)
Settled-boundary rule (every cross pair >= 0.95), BEST scope:
 his page: 13 stages, sizes 10,1,7,3,2,4,1,2,8,4,1,2,5. Combined uncertainty: 7 stages, sizes 10,8,20,4,1,2,5.
 Combined stages: (1) 1a x10; (2) 1b x8; (3) ONE unresolved block of 20 = all of 1.5a, 2a, 3a, 3b, 5a, 6; (4) 3c x4 (N:85-86 + T sheets); (5) T:107-112; (6) 4c x2; (7) 4a bio x5.
 Decisive pairs: all 0.897 (his 0.923); same-group 0.222 (his 0.389). ALL scope (every start): 3 stages 10, 8, 32 (his ALL: 6).
 Threshold sensitivity: 0.90 -> 10,8,5,5,10,4,1,2,5 (middle splits in three); 0.99 -> 18,32. 1a / 1b separation holds from 0.80 to 0.95.
Reading (descriptive): with all uncertainty combined, his order resolves into ~7 stages; the 1a -> 1b -> {middle} -> late-B blocks are settled; the internal order of the
 middle 20 bifolia (1.5a through 3b incl. pharma/astro) is NOT settled. P(survives 60d): "1a, 1b and the middle block separate" 80%; "middle internal order unsettled" 85%.

## 2026-10-17 ~23:59 — Ed correction on PREREG 15 (edge-soiling test)
Ed: f116v is NOT stained (contra Stolfi). PREREG 15 used f116v as one of two positive controls ("known board/leather discolouring"); it ranked only 66th percentile,
which made the instrument VOID. If f116v is not stained, that control was invalid and the VOID was for the wrong reason (f116v's 66th percentile is then what an
unstained page should show). Rule 18: the gate cannot be rewritten after the fact, so the unregistered numbers (interior-sheet D z 2.37, 18/29) still carry NO verdict.
Status: PREREG 15 -> NEEDS RE-RUN with a valid positive control (f1r passed at the 97th percentile; a second known-discoloured page must be chosen BEFORE re-running).
Images: 162 IIIF page photos already on disk (horseshoe/wear/img, 1000 px); Yale manifest reachable (HTTP 200).

## 2026-10-18 ~00:10 — Mauro (voynich.ninja) comparison of npcompl33t's order with Layfield & Davis 2026 (Digital Medievalist 2331, "Singulion Structure") — assessment
Source check: Layfield & Davis sequence bifolia by LSA text similarity between adjacent pages (abstract via doi.org/10.4000/16k0a); same TEXT as npcompl33t -> agreement is two
 text-similarity methods agreeing, NOT independent physical validation. Timm's review (academia.edu 170072857) notes their resequencing lacks a permutation null and a
 shuffled-known-text calibration. mgvtheus (voynich.ninja thread 5940) re-ran their method on ZL3b: Q20 matched, Q13 edges changed -> Q13 order unstable across transliteration/algorithm.
 Davis's one physical-ish cue: f76r and f105r are the only pages in Q13/Q20 that START with an enlarged glyph (her post, thread 5896).
His published positions (ZL, l .01, best-scope medians): page-level Q20: f105 0.836-0.841 < f104 0.869-0.871 ~ f106 0.871 < f107 0.873/0.906 < f108 0.911/0.949 ~ f103 0.950;
 our combined-uncertainty bifolium stage map: {105,104,106} < 107 < {108,103} -> exact match is partly inside unsettled ties; coarse Q20 order and 105-first are robust.
 Q13: pool50 all five bio bifolia = 1.000; page-level 17/20 bio pages 0.995-1.000, 13 at exactly 1.000 (curve endpoint). His Q13 "sequence" comes from three pages
 (f79r 0.983, f76r 0.988, f82r 0.992) barely off the endpoint -> NOT resolved. Chance of a 5-item order matching another up to reversal + one adjacent swap ~ 10/120 = 8% (my arithmetic, uniform-order assumption).
 f1r: median 0.015 but 5-95% span 0.000-0.313, rank 18/207 -> its place within early 1a is not resolved.
 Bio after f116r: bio pinned at the curve endpoint and confounded with scribe/section; Mauro's surprise is justified; "appendix" reading not supported by this method.
Read Layfield & Davis 2026 (Digital Medievalist 19, PDF supplied by Ed): ZL v2a (17/03/2022), uncertain readings removed, heuristic stop words removed, 40-word chunks,
 TF-IDF + LSA, cosine. Resequencing (para 50-54): EXHAUSTIVE search of all 120 (Q13) / 720 (Q20) sequential orders maximising mean facing score; NO permutation null reported.
 Q13 best: 77|82, 78|81, 75|84, 76|83, 79|80 (+12.1%); second-best 76|83, 77|82, 79|80, 75|84, 78|81 (+7.1%), which they lean towards on visual + enlarged-glyph grounds
 -> their own top two Q13 orders disagree almost completely: Q13 order not determined by text even in their method. Q20 best: 105,104,106,107,108,103 (+20%).
 Note: best-of-720 always beats the current order; the % improvements are uninterpretable without a null (Timm's critique).
 Q20 agreement with npcompl33t: his groups 3c = {105,104,106,107}, 4c = {108,103}; 48/720 orders put 3c before 4c, so beyond the group split the internal match is 1/48 by chance
 (uniform assumption, my arithmetic) — and his internal Q20 positions are not settled under resampling. Same transcription family (ZL) in both; word-level vs glyph-transition features of the same text.

## 2026-10-18 07:30 — Supabase sync: handoff_docs voynich_horseshoe_programme_20261018 (10,391 chars) and voynich_npcompl33t_order_stage_map_20261018 inserted (research-agent project). Local copies in handoff/. Nothing else in Supabase changed.
