## RETRACTED / CORRECTED FINDINGS
- 2026-11-14 CORRECTED after reading eo777's repo docs: "his four page paths disagree (99-114/205 shared), so the page order is unstable".
  Paths 2-4 are DESIGNED conditional variants (different 3a/3b and 4a/4c ending constraints), not replicates.
  Their disagreement measures sensitivity to those constraints, not noise. The claim "unstable at page level" is downgraded to
  "more sensitive to the ending constraints at page level than at bifolium level".
- 2026-11-14 RESOLVED uncertainty: page and bifolium scores are SEPARATELY fitted cohorts (207 pages; 50 pooled bifolia), so the
  closeness of sheet partners in the page path (F1c) is not built in by deriving one from the other. (Pages are weighted equally per sheet.)

# Bifolium ordering consensus (2026-11-14) — DESCRIPTIVE, not pre-registered
Inputs: eo777 golf-v21 (4 bifolium paths, 50 sheets), ZLZI word cosine (Voynichdecomp slim json).
- Golf paths agree with path 1 on 33/38/40 of 49 adjacencies; 29 adjacencies appear in all four.
- Independent text seriation (word cosine, greedy + 2-opt shortest path) vs Golf consensus: Spearman |rho| = 0.87;
  shares 24/49 adjacencies with path 1 (chance about 2). 15 of the 29 all-four pairs are also in the text path.
- Interpretation: Golf is reproducible from plain word overlap, so it is a TEXT-SIMILARITY SERIATION.
  Agreement = method robustness, NOT evidence of physical order (both inputs are text).
- Calibration (Oct 2026, horseshoe s28): on known-order texts at VMS sizes, shortest path recovers ~21% of true
  neighbours (chance 4%). Expect roughly 1 in 5 consensus neighbours to be true physical neighbours; which ones is unknown.
P(text seriation at VMS size recovers <=30% of true neighbours) = 70% (carried over from October).

## VoynichStats within-quire optimiser, 6 settings (cos/tfidf/jaccard x word/bigram), ZLZI, 500 runs, seed 42 — DESCRIPTIVE
See quire_stability.csv. Caveat: the suggested order is the best of n! nestings (24 for 4 sheets), so "beats 9x% of random"
is by construction. Informative = where the RECORDED order sits (beatRecorded) + agreement across settings.
- Recorded order already best/near-best in all settings: C (0.92-0.96), E (0.66-0.95). Expected by chance: ~1/24 per 4-sheet quire.
- Recorded order consistently poor: A (0.03-0.41; 4/6 settings pick 3-1-2-4).
- Stable re-nests (>=4/6 settings): A 3124, F 1342, G 2341; H and S swap 6/6 (2-sheet quires, tiny deltas for S).
- No consensus: B, D (3/6), M, T (3/6).
- Settings are NOT independent (all text-similarity metrics on the same words).
order_impact for {A 3124, F 1342, G 2341, H 21, S 21}: pageJaccard 0.1100 vs recorded 0.1082 vs random 19 [0.1057-0.1097]
-> above all 19 random, but CIRCULAR (the optimiser maximised similarity). MATTR500 0.6039 vs recorded 0.6058 (random 0.6027-0.6061);
lag1 unchanged (0.1599 vs 0.1598). No independent support.
P(any of these within-quire re-nests is the true original nesting) = not estimable without a physical anchor.

## PRE-REGISTRATION F1 (2026-11-14 ~06:25, before computing): folio-level checks on Golf page paths
Data: golf-v21-assignments.csv page_path_rank / _2 / _3 / _4 (227 pages). Independent comparator: ZLZI word-cosine
shortest-path seriation of pages (greedy + 2-opt), and of leaves (recto+verso).
F1a PHYSICAL CHECK: share of leaves whose recto and verso are adjacent in the Golf page path. Null: 10,000 random page
permutations. A physical reading/writing order should keep r/v adjacent (expected well above 50%).
  >= 50% -> Golf page path behaves like a physical order. <= 2x the null mean -> it does NOT (text-similarity order only).
F1b Agreement between the Golf page paths and the text-only page path: shared adjacencies and |Spearman|.
F1c Bifolium coherence at page level: mean |rank gap| between the two leaves of a sheet vs the null.
F1d Where the crease leaves (f5, f15, f21, f29, f32, f37) and their partner leaves (f4, f10, f20, f25, f28, f36) sit in the
page path: span of their page ranks vs a matched random set of leaves of the same size, drawn from quires A-H.
Prediction: F1a fails (P=0.7); F1b strong agreement (P=0.8); F1d crease leaves clustered (P=0.6), partners also clustered (P=0.6).

## F1 RESULT (pre-registered), 206 pages with page-path ranks (fRos excluded; 92 leaves with both r and v)
F1a r/v adjacency: Golf page paths 17.4% / 13.0% / 10.9% / 14.1% vs random 1.2% ± 1.1%. Not >=50%, not <=2x null ->
  the pre-registered rule had a GAP here (my design fault). Plain reading: far above chance (r and v share text), but 83-89%
  of leaves have r and v SEPARATED, so the page path is NOT a reading/writing order.
F1b Golf page paths share only 99/114/111 of 205 adjacencies with path 1 (vs 33-40/49 at bifolium level).
  Text-only page seriation shares 20/205 with Golf path 1 (chance 2), |Spearman| 0.14 (vs 0.87 at bifolium level).
  -> Golf is stable and reproducible at the BIFOLIUM level only; at page level it is unstable and not reproduced by word overlap.
F1c Conjugate leaves: mean rank gap 6.0 vs null 70.3 ± 6.4 (z 10.1, p 1e-4). Could be partly built in (unknown whether
  his page path is constrained by sheet); consistent with Oct 2026 "conjugate leaves share a text state".
F1d Crease leaves span 24 page ranks vs 114 ± 26 (z 3.48, p 0.0009); partner leaves span 32.5 (z 3.15, p 0.0038).
  Uniform null, not depth-matched. Partners cluster as well (F1c makes that automatic) -> leaf level adds nothing
  beyond the bifolium result. Damaged-before-partner in page order: 4 of 6, after 1, tie 1 -> the B side is not ordered.
P(survives 60d): "Golf ordering only meaningful at bifolium level" 65%.

## eo777/voynich-clusters repo read (commit 778066a), 2026-11-14 — docs read in full, code not read line by line
Method: Zattera glyph-unit transition matrices (29x29; word + joined-line), row-weighted JS divergence, classical MDS (all positive
axes), ElPiGraph elastic principal curve (28 nodes, mu 0.1, lambda 0.01/0.05), 100 x 90% subsamples x 8 starts, ZL + RF;
score = pairwise 'later than peers' consensus. DESIGN INPUTS (his words): starts pass through Golf group centres; direction set as
Early A -> Bio/Stars; 1a-before-1b filter. 'Not dates, not chronology probabilities.'
His own findings: 3a/3b order unsettled (wins 19-58/100 by setting); Stars 4c before Bio 4a usually; endpoint bifolium identity unstable;
f58 NOT robustly a discontinuity (he withdrew his own 'pocket' reading after calibration); low detection power stated.
v1: within-bifolium variation along path 0.758% (held-out 1.549%) -> matches our F1c.
Unread: fit.py/core.py/models.py and other code line by line; DESIGN.md files; HTML reports; b_horseshoe, topology, influence code.

## L1 RESULT (pre-registered): his label-free fit vs published Path 1 (50 bifolia, ZL, 100 samples x 8 starts x 2 penalties)
|Spearman| = 0.912 (paths 2/3/4: 0.850 / 0.904 / 0.908); Kendall 0.773; 29/49 adjacencies shared.
-> >= 0.90: the overall bifolium order does NOT depend materially on Golf-label starts or filters.
BUT the Early-A segment is rearranged: largest moves are all Herbal A (A:1-8 2->18, D:27-30 1->14, F:42-47 3->17,
B:12-13 16->1, B:11-14 15->2, C:19-22 14->3). The internal Early-A order is where his design inputs (1a-before-1b, 'Early A return') act.
Secondary: BOTTOM-N6 on the label-free order: span 11 vs 16.19 ± 2.85, z = 1.82, p = 0.054 (published order: span 7, z 3.25, p 0.003).
Inconclusive at this power (< 2 null SD). The crease set's tightness is partly a product of the label-conditioned Early-A arrangement.
P(survives 60d): 'global order label-independent' 75%; 'N6 adjacency is robust' revised to 25%.

## eo777 other repos read 2026-11-14 (READMEs and key docs in full; code and the 175 KB mattr RESULTS.md NOT read)
GitHub profile lists 11 public repos; I could see 6 names + 2 known = 4 Voynich repos: voynich-clusters, voynich-rules (Echo A->B spelling
rules, app only, 18 Oct), vms-dynamic-citation (29 Oct), vms-spatial (6 Nov). The other 3-5 names are unseen (API blocked).
vms-spatial: word rectangles (Voynichese Project) + ZL, dewarping, ink baselines on Beinecke scans, page-overlay comparison, Light table
 (lay bifolia out as sheets/leaves, stack as a quire), sheet/fold audit (docs/SHEETS.md), MSI margin check f1r/f8r.
 His own statements: 'Path 1 ... is not an established chronology'; 'shared curvature is not identified as binding: its stronger side
 falls on the gutter side on about half of the pages, which is chance'; scan scale 15.84 px/mm assumes 225 mm leaf height.
 NO crease / tail-edge / damage model in the repo as of 6 Nov (grep): the thread-6299 damage work is unpublished.
vms-dynamic-citation: Timm self-citation generator steered by a positional model fitted along Path 1; MATTR by bifolium; spatial
 'source pool' diversity vs repetition. He corrected his own headline on 28 Oct: raw +6.1-point contrast mostly a word-length artefact;
 residual +2.6 (components) / +3.5 (H2) above 19 length-matched shuffles; absent in Herbal B; 'not diagnostic of copying'.
 DEPENDENCY: this work reads the text in Path 1 order, so anything order-sensitive there inherits the Path 1 design inputs.
