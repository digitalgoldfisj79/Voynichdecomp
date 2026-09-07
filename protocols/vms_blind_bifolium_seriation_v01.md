# VMS blind bifolium seriation v0.1 — FROZEN BEFORE TARGET EDGES

Date: 2026-09-07
Protocol ID: `vms_blind_bifolium_seriation_20260907_v01`

## Question
Can independently measured bifolium production states recover reproducible local production neighbourhoods without using present folio order, current adjacency, quire position, section, Currier or hand as ordering inputs?

## Non-negotiable separation
This assay does **not** seek or licence a unique total folio order. Its target is an undirected partial neighbourhood graph. Direction/order is only licensed later where independent physical evidence supplies orientation.

## Leakage controls
The graph optimiser may use only opaque node IDs generated from unit ID + fixed salt. It must not use folio number, current page order, current quire position, section, Currier, hand, illustration category, current adjacency, stain/sewing evidence, or any previous proposed Voynich ordering.

Current codicological/metadata labels are revealed only after candidate edges are frozen.

## Nodes
All surviving physical bifolia with text on both leaves in the frozen explicit conjoint registry. Missing and partial bifolia remain topology nodes elsewhere but do not enter text seriation v0.1.

## Input representations
Four independently maintained IVTFF transcriptions, exact source hashes inherited from topology marginalisation: ZL, RF, IT, GC.

## Discovery feature families
Each family is scored independently.

1. `LEX` — word-frequency Jensen-Shannon distance, vocabulary selected by global count threshold only.
2. `C3` — within-token boundary-marked character-trigram Jensen-Shannon distance; no across-token or across-page ngrams.
3. `SHAPE` — token-length + initial-character + terminal-character distributions; no folio/order metadata.

The three families are deliberately redundant enough to measure production state but not identical representations.

## Held-out validators
Not used to select candidate edges:
- `RARE`: weighted/Jaccard overlap of globally rare word types.
- `LAYOUT`: line-token-count distribution and page/line density where recoverable.
- IVTFF `$L/$H/$I`: Currier/hand/broad section, reveal-only confound audit.
- present folio adjacency / present quire: reveal-only historical comparison.
- hard physical topology constraints: reveal-only compatibility audit.

## Edge construction
For every transcription × discovery family:
- compute all pairwise distances between opaque bifolium nodes;
- each node nominates its two nearest neighbours (`k=2`);
- a channel edge is supported only when nomination is reciprocal.

A frozen consensus candidate requires:
- support in at least 6 of 12 transcription×family channels;
- support from at least 2 of 3 feature families;
- support in at least 3 of 4 transcriptions.

No Hamiltonian/TSP/path optimisation is permitted.

## Stability
For every consensus candidate, run 128 within-node bootstrap resamples per transcription. Report channel/consensus recovery frequency. `stable_edge` requires bootstrap consensus >=0.60.

## Accidental-consensus null
Independently permute node labels for C3 and SHAPE relative to LEX, preserving each family distance matrix, 5000 times. Recompute consensus edge count and largest component size. This tests whether cross-family agreement is greater than chance without altering marginal clustering.

## Known-answer calibration
Before interpreting bifolium-to-bifolium edges, apply the same blinded multi-family nearest-neighbour logic to folios in exchangeable strata where the true conjoint partner is independently known. Evaluation labels are revealed only after matching. Report precision/recall/top-2 recovery. Failure blocks ordering inference but does not erase the already-qualified bifolium-unit effect.

## Reveal-stage confound audit
After edges are frozen:
- reveal current folio IDs, quire, Currier, hand and broad section;
- classify whether each edge simply joins identical metadata strata;
- compare candidate scores with all alternative pairs in the same revealed metadata-pair class;
- no edge is promoted as a production-neighbour edge solely because it agrees with current adjacency or current quire.

## Partial-order interpretation
Connected components are reported as undirected neighbourhood components. A component may be rendered as a path only if every node degree <=2 and edge stability passes. Path reversal is equivalent. Branching components remain unordered graphs.

## Decision language
- If known-answer calibration fails: `SERIAL NEIGHBOURHOOD INFERENCE NOT QUALIFIED`.
- If candidate count is not above accidental-consensus null by >=2 SD: `the metric does not resolve a production-neighbour graph`.
- If a stable edge survives reveal-stage matched-confound audit: `candidate production-neighbour edge`.
- Never call text-derived edges `original order` without independent physical orientation evidence.
