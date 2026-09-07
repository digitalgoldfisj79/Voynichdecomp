# VMS pure-text bifolium connection programme v0.1

Protocol ID: `vms_textual_connections_20260907_v01`
Frozen: 2026-09-07

## Retraction / scope

The RGB stain / physical-image ordering branch is excluded from this programme. No stain, offset, parchment colour, image feature, current folio adjacency, current quire position, quiremark, hand, Currier language, section, illustration category, or other codicological metadata may enter target discovery or scoring.

The qualified result that physical bifolia are unusually coherent lexical-production units remains in scope because it was established textually under matched controls. Whole-bifolium similarity is NOT sufficient evidence for connection and is prohibited as the primary connection score.

## Question

Can two bifolia be connected by a signal that is specifically concentrated at textual boundaries, rather than by whole-document state similarity, and can any such signal support direction?

A `TEXTUAL_CONNECTION` means: one textual port of unit A has a calibrated, boundary-specific relationship to one textual port of unit B that is stronger than A/B interior similarity and stronger than matched alternative partners. It is not automatically a physical adjacency or chronological claim.

## Target sealing

Target bifolium IDs are opaque during scoring. Folio numbers and metadata are revealed only after candidate edges, ports, directions, null statistics, and bootstrap stability are frozen.

Complete bifolia only for v0.1. Missing/partial units are excluded rather than imputed.

Four frozen IVTFF corpora: ZL, RF, IT, GC with existing project SHA-256 verification.

## Textual ports

Each page is ordered only internally by its transcribed line/token order, which is topology-safe. For each page surface create:

- `HEAD_K`: first K word tokens.
- `TAIL_K`: last K word tokens.
- K = min(64, max(24, floor(page_tokens/3))).

A bifolium is treated as four page surfaces and therefore up to four HEAD ports and four TAIL ports. No current page-to-page order inside the bifolium is assumed. A directed A→B connection is the best TAIL(A_surface)→HEAD(B_surface) bridge, with the max operation reproduced identically in every null.

## Frozen feature families

### F1 RARE-BRIDGE
Global corpus token frequency is computed without topology. Tokens with global frequency 2–12 are weighted by inverse log frequency. Score a TAIL→HEAD boundary by weighted overlap divided by weighted union. Whole-unit overlap is not used.

### F2 SUBWORD-PREDICTIVE
Build character 3/4-gram distributions from TAIL and HEAD windows with boundary markers. Score TAIL→HEAD using symmetric Jensen-Shannon closeness plus a directional cross-entropy transfer term: an add-alpha n-gram model fit on TAIL predicts HEAD. Alphabet/vocabulary is frozen corpus-wide, not candidate-specific.

### F3 MORPH-PORT
Represent each window by token-length distribution, initial 1/2-char distribution, terminal 1/2-char distribution, and the project's topology-safe token component decomposition where available. Score closeness with Jensen-Shannon distance. No hand/Currier/section labels.

No additional family may be added after target opening in v0.1.

## Boundary-specificity control

For every candidate A→B and every feature family, compare the winning TAIL(A)→HEAD(B) score against the same pair's non-boundary controls:

- HEAD(A)→HEAD(B)
- TAIL(A)→TAIL(B)
- HEAD(A)→TAIL(B)
- interior-half(A)→interior-half(B)

The exact same max-over-surfaces operation is used. A connection cannot pass solely because two bifolia share a broad lexical state.

## Alternative-partner null

For the winning source port of A, compare B with all other eligible bifolia under the same corpus. Candidate identity and metadata are not used to select alternatives. Because the target graph contains strong section/state clusters, a second post-reveal diagnostic residualises the score against hand/Currier/section/illustration metadata, but this diagnostic may only DELETE a candidate, never create or strengthen one.

## Calibration BEFORE Voynich reveal

### C1 continuous-text pseudo-singulions — Caesar
Use frozen `caesar_dbg_i_iv_gutenberg_218`, SHA-256 `976c130b6637c8b643617e66e039b8fddcb610eda02c3b68db1bdce87bd09866`.

Create page chunks with token lengths sampled deterministically from the empirical VMS page-length distribution, preserving Caesar order. Group each four consecutive chunks as a pseudo-singulion. Hide order and attempt to recover the true boundary between pseudo-singulions using the exact target algorithm. Use many sliding windows and deterministic seeds.

Qualification gate for a family:
- true adjacent pseudo-unit edge recall >= 0.70;
- false-edge rate <= 0.20;
- directional accuracy among recovered true edges >= 0.70;
- true-boundary score >=2 matched-null SD in >=75% of windows.

The combined method qualifies only if >=2/3 families qualify and the consensus graph has recall >=0.75 and false-edge rate <=0.15.

### C2 historical page-order control — Aberdeen Bestiary MS 24
Use the 32 already-frozen folio-page Latin transcriptions, ff.80r–95v, with per-page SHA-256 values stored in `vms_topology_control_pages_v01`.

At page level, hide page order and rank each page's true successor within its quire using the same TAIL→HEAD families. Report top-1 successor accuracy, top-3 accuracy, mean reciprocal rank, direction accuracy, and permutation null.

Gate: combined top-3 successor accuracy >=0.60 and observed top-1 count >=2 matched-null SD. Failure does not necessarily kill C1, but it prohibits calling the method historically manuscript-qualified; target outputs would remain exploratory.

## Target candidate rule

Only if C1 combined qualification passes may the Voynich target be opened.

An undirected `TEXTUAL_CONNECTION_CANDIDATE` requires:
- consensus support in >=3/4 transcriptions;
- support from >=2/3 feature families;
- bootstrap edge stability >=0.70;
- winning boundary score >=2 alternative-partner null SD in >=3/4 transcriptions;
- boundary-specificity advantage >=2 within-pair null SD in >=3/4 transcriptions;
- no deletion failure under leave-one-family-out or leave-one-transcription-out consensus;
- post-reveal metadata residualisation does not delete the edge.

Direction A→B is separately licensed only if:
- C1 directional accuracy gate passed;
- A→B beats B→A by >=2 direction-null SD in >=3/4 transcriptions and >=2 families;
- directional bootstrap stability >=0.70.
Otherwise edge remains undirected.

## Decision language

If an effect is <2 matched-null SD: `the metric does not resolve this`.

Do not infer physical adjacency from a textual connection without independent evidence. Do not infer chronology from an undirected connection. Do not assemble a total order unless independently qualified directed edges naturally form a path.

## Stop rules

- If C1 combined qualification fails: do not open Voynich target; redesign only in a new preregistered protocol.
- If C1 passes but C2 fails: target may be opened only as exploratory text topology, not manuscript-order inference.
- If no target edge passes: close v0.1 negative. No threshold relaxation.
- No stain/image/physical result can rescue a textual edge in this protocol.
