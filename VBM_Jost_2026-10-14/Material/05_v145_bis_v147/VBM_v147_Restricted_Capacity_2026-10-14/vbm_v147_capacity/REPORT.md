# VBM v147: capacity-first, scribal constrained audit
14 October 2026 · **Restricted architectural NO-GO; no historical or plaintext attribution**

## Outcome and limits, before interpretation

1. **Mathematical theorem under specified assumptions.** No bijection from the 23 German source letters to single Voynich glyph atoms can make every **observed** source trigram map into the **attested** A or B atom-trigram support, either as a position-independent alphabet or as v146's three fixed initial/medial/final alphabet columns. This holds for both development text and already reused Hartlieb diagnostic text. It is a pigeonhole bound, independent of optimisation choices.
2. **Scope of theorem.** Requires one source character -> exactly one visible EVA atom; output length in atoms equals source length; decoding uses fixed bijective columns. A later stateful transducer, variable-length abbreviations, unseen permitted Voynich triples, or a different source language can evade it. The sample transcription's finite support is not the complete historical grammar.
3. **Not yet done.** This is not a capacity test of the canonical October 4 K8/K12 FORM graph (~32 pieces /12 classes/~105 transitions). That graph, its exact transition rules, stop permissions and section-conditioned state model were not reconstructed here. We **must not** rename the observed-trigram filter “FORM” or claim the new restricted theorem retires all short stateful encoders.
4. **Scribal gate not met.** No demonstrated manual source parser, <=few-table decoder, physically natural strokes, or predictive scribal error signature. There is no manuscript mechanism identification.

## Fixed evidence and full code

Inputs are exact archived v146 working references, copied without mutation into `in/`. Their SHA-256 hashes appear in `RESULTS.json`. Native words use the original v146 REVA reversal-of-compound-atoms convention: `ckh|cth|cph|cfh|ch|sh|.`. The two reference vocabularies are pooled A and B; B lines contain line IDs before the tab. Normalized German sources contain 68,396 development words (8,247 types) and 13,232 previously examined Hartlieb words (2,276 types). **Neither German source is evidence that Voynich's hidden language is German.** A and B sources are already used as fitting references in v146; this audit does not claim blind folio validation. Alternative transliteration and hand/section sensitivity were not run.

## Exact distinct triple inventories

| Unit triple category | Development German | Hartlieb German | Native A | Native B |
|---|---:|---:|---:|---:|
| Any | 2,959 | 1,750 | 1,255 | 1,485 |
| Start I-M-M, words with >=4 atoms | 1,166 | 703 | 438 | 528 |
| Interior M-M-M, three genuine medial positions | 2,520 | 1,310 | 646 | 834 |
| End M-M-F, words with >=4 atoms | 990 | 535 | 650 | 763 |
| Exactly 3-atom words I-M-F | 398 | 181 | 279 | 315 |

For a bijection in each of three positions, distinct input `M-M-M` triples must have distinct output `M-M-M` triples. Consequently **at least 2,520−646=1,874** distinct development-source medial triples fail observed A support, and **at least 2,520−834=1,686** fail B support. On Hartlieb the corresponding necessary lower bounds are **664** A and **476** B. These are counts of distinct triple *types*, not numbers of words or error percentages. Bound holds even when allowing **every** attested reference triple regardless of whether its atoms are among the 23 table signs (an optimistic support envelope). The start-triple bounds also fail independently. No bijective permutation can repair this within the frozen one-atom scheme.

For a single shared substitution table, the any-position pigeonhole is 2,959 vs 1,255 A / 1,485 B on development. The corresponding Hartlieb bounds are 1,750 vs 1,255/1,485.

## Useful counterweight: don't mistake that for general capacity failure

The Hartlieb source contains 2,276 distinct word types, fewer than the *observed* A 3,098 and B 4,778 native word types. A hypothetical unrestricted 2,276-row word dictionary could assign a unique attested word to every Hartlieb type and decode it exactly. That construction would be completely target-derived, historically unexplained and scribally expensive. Development has 8,247 types, more than the frozen exact-word lists. Neither result is a general finite-state capacity proof, because unobserved forms and source constraints could change the counts.

**Conclusion:** scarcity of attested *local one-atom transitions* rules out v146's fixed bijective letter tables, while sheer whole-word label capacity does **not** rule out a giant arbitrary dictionary for the small held-out source. The unresolved challenge is **compact compositional, uniquely decodable rules with a realistic manual implementation**.

## Frozen next admissibility test (not performed in this package)

- Recover the **actual frozen Oct 4 FORM/K12** piece/classes/transitions/stop gates with provenance, not a newly inferred grammar or naive pooled trigrams. Compare native A/B as independently sectioned running text.
- Bound source event payload and legal full-word path counts under FORM by **word length and line context**, treating multiple permitted strings that decode to the same value as redundancy, not capacity. Derive lower and upper bounds before codebook fitting.
- Define a small self-delimiting deterministic transducer with <=3 observable working states and <=69 literal table entries, as a *precommitted exploratory budget* (not a historically attested parameter), plus at most 10 rules. Declare what the scribe reads and writes for a word, how he resets at lines, and how a separate decoder recovers it. No manuscript-derived per-word dictionary; no off-page side channel; no complex optimization carried out by the scribe.
- Test synthetic planting and matched negatives *before* Voynich application; use independent native folio/quire holdout for word-form legality, length tails, section specificity and attested positional patterns, not only training-derived trigram legality. Report complexity/lookup effort, source-text lengths, and handwriting ductus separately.
- Do **not** call successful reversible encoding a historical Voynich cipher or interpret `v143` visible state as plaintext carry without an external anchor.

## Reproduction

`python run_capacity.py`

The script is stdlib-only and reads `in/` relative to its own path. Prints inventories and writes `RESULTS.json`. Run in a fresh extracted directory as a direct replay; the key counts are assertion-checkable by recomputing distinct sets. `RESULTS.json` reports all SHA-256 hashes. This package includes exactly the archived v146 inputs used for the claim.
