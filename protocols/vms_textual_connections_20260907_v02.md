# VMS pure-text connections v0.2 — page-first adjacency kernel

Protocol ID: `vms_textual_connections_20260907_v02`
Frozen: 2026-09-07

## Retractions / relation to v0.1

- RGB stain/image evidence remains excluded completely.
- v0.1 TAIL→HEAD pseudo-bifolium recovery is CLOSED NEGATIVE: combined recall .114, false-edge .791 on canonical frozen Caesar. It may not be retuned or applied to Voynich.
- v0.2 does not discover bifolium edges by maxing over 16 surface pairs. It discovers page-surface edges first under independently qualified scalar thresholds; mapping to physical bifolia happens only after page-edge freezing.
- Whole-bifolium similarity cannot create a v0.2 edge.

## Question

Does Voynich text contain a generic local-continuity signal that can identify which textual surface is connected to which other surface, independently of current codex order and metadata?

`TEXTUAL_ADJACENCY` is an operational relation: two text windows exhibit the kind of local continuity that a frozen kernel recovers in known ordered text. It is not automatically physical adjacency or chronology.

## Stage M — topology-safe Voynich micro-calibration

Use only ordinary paragraph-formatted lines and their internal line order. This ordering is intrinsic to a page and does not depend on binding topology.

Parse IVTFF paragraph runs from line flags (`@P`/`*P` start, `+P` continuation, `=P`/`=Pt` termination); exclude labels, circular text and non-P loci. Admit paragraphs with >=8 nonempty lines.

Split each admitted paragraph into non-overlapping two-line blocks. A positive pair is adjacent blocks `(i,i+1)`. Negatives are nonadjacent block pairs from the SAME paragraph. Therefore hand, Currier, page, illustration, topic and transcription are exactly matched by construction.

Frozen generic feature vector, no token-identity coefficients:
1. WORD_COS — word-frequency cosine over complete blocks.
2. RARE_JACC — globally frequency-weighted overlap of token types occurring 2–20 times in that transcription.
3. CHAR3_JS — negative Jensen-Shannon distance of within-token character trigrams.
4. MORPH_JS — negative JS distance over token length + first/last 1/2-character features.
5. BOUNDARY_WORD — cosine between last line of A and first line of B minus mean of A-first/B-first, A-last/B-last, A-first/B-last controls.
6. BOUNDARY_CHAR — same boundary enrichment using char-trigram similarity.
7. EXACT_BRIDGE — fraction of token types in last line of A recurring in first line B, baseline-corrected by the three non-boundary line pairings.

Fit an L2 logistic regression only to these seven scalar features. Cross-validation is GROUPED BY PAGE: no page contributes pairs to both train and test. Standardization is fit on training folds only. Use deterministic 5-fold GroupKFold independently for ZL/RF/IT/GC.

Stage-M family qualification per transcription:
- held-out ROC AUC >= .70;
- held-out average precision >= 2.0 × test-fold positive base rate;
- adjacent-pair mean predicted score exceeds within-paragraph label-permutation null by >=2 SD.

Stage M passes if >=3/4 transcriptions qualify. If it fails, STOP v0.2; do not open any page-scale control or target.

## Stage C — page-scale known-order transfer

Run only if M passes. Kernel coefficients are refit on all eligible Voynich micro-pairs within each transcription; no page-order target information is used.

### C1 Caesar page successor control
Use canonical frozen Caesar main body SHA-256 `1135e90a23cc19460e5827cec1b9dddf8b1aeaddf8b598b1dcc06b6f8c2255d3` and deterministic page lengths from the frozen VMS empirical page-length distribution.

Create a sequence of pages. For every page i except the last, rank every page j in the same 16-page window except i by the learned generic adjacency kernel. Score is undirected for v0.2; success means true immediate neighbour i-1 or i+1 is recovered. Use sliding 16-page windows.

C1 pass:
- mean top-1 adjacent-page accuracy >= .35;
- mean top-3 adjacent-page accuracy >= .65;
- observed top-1 true-edge count >=2 permutation-null SD;
- false mutual-edge rate <= .35 for mutual-top1 graph.
Require >=3/4 transcription-trained kernels to pass.

### C2 Aberdeen Bestiary historical page control
Only if C1 passes. Use already frozen ff.80r–95v Latin transcriptions / hashes in the project or deterministic public-page retrieval verified against stored token counts.
Within each intact quire M/N, rank page neighbours with the same frozen generic kernel; no retraining on Aberdeen.

C2 pass:
- top-3 immediate-neighbour accuracy >= .50;
- top-1 true-edge count >=2 permutation-null SD.
If C2 fails, Voynich target remains sealed for manuscript-scale adjacency. A Caesar-only success is insufficient.

## Stage T — Voynich page graph

Only if M+C1+C2 all pass.

Target page IDs are replaced by opaque IDs before scoring. Current page order, folio number, bifolium, quire, hand, Currier, section and illustration metadata remain sealed.

Each page gets one full-page representation plus boundary line representation. Score all page pairs with the same trained generic kernel independently in ZL/RF/IT/GC.

A page edge is frozen if:
- mutual top-1 in >=3/4 transcriptions OR mutual top-2 in 4/4;
- pair score >=2 alternative-page null SD in >=3/4;
- bootstrap stability >=.70 in >=3/4;
- leave-one-feature-out: removing any single scalar feature does not reduce support below 3/4.

Only after page edges are frozen reveal physical bifolium membership. Collapse cross-bifolium page edges into unit edges. A bifolium connection requires at least one frozen page edge; two independent page edges between the same bifolia are reported separately as stronger evidence but do not change thresholds.

Metadata residualisation (hand/Currier/section/illustration/current quire) is post-reveal and DELETE-ONLY. Any edge disappearing after residualisation is state-confounded and cannot be called a production-neighbour candidate.

## Direction

v0.2 is primarily UNDIRECTED. Direction is not inferred unless a separate preregistered directional kernel passes >=.70 direction accuracy on both Caesar and Aberdeen. No direction model is part of the initial v0.2 opening.

## Decision language

|effect| <2 matched-null SD => `the metric does not resolve this`.
No total order is assembled from isolated edges. A multi-edge component is reported as an undirected path/component unless direction is independently qualified.

## Stop rules

M fail => stop before page controls.
C1 fail => stop before Aberdeen/target.
C2 fail => target remains sealed.
T no edges => close negative; no threshold relaxation.
No physical/image evidence can rescue an edge.
