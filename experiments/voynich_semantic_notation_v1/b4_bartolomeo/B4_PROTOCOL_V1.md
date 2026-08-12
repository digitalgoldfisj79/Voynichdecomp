# VSN-B4-v1 — Bartolomeo Attested Notation Programme

Frozen: 2026-08-12 Europe/London
Parent: VSN-v1 Workstream B
Branch: `experiment/vsn-b4-bartolomeo-notation-v1-20260812`
Base commit: `bc4a9b1324cb428bd0431a46bb360dbd2aad7618`

## Scientific question

Does Bartolomeo da Mantova's *Liber memoriae artificialis* (1429), using only notation actually attested in the manuscript, exhibit the structural constraints that the synthetic Matteo/state-gated models failed to reproduce: strong transition constraint, positional/right-edge asymmetry, dense local morphology, and context/position-dependent interpretation?

This is a historical mechanism test. It is not a decipherment claim and does not assign Bartolomeo's meanings to Voynich glyphs.

## Primary-source core

Paris, BnF, Latin 8684, Gallica ARK `btv1b520004421`.

Frozen folios supplied from the Gallica witness:
- f.7r — Gallica image sequence f17
- f.7v — Gallica image sequence f18
- f.8r — Gallica image sequence f19
- f.8v — Gallica image sequence f20
- f.9r reported blank by direct inspection; not an analytical folio.

The supplied 1024x1387 JPEGs are source-locked by SHA256 in `PRIMARY_SOURCE_MANIFEST.csv`.

Primary-source interpretation has priority over secondary summaries. Illegible readings remain unresolved.

## Secondary source

Valentina Cacopardo, *Memory and Imagination in the Ars Memorativa in Fifteenth-Century Italy* (PhD thesis, School of Advanced Study, 2021), especially pp. 135–136 of the PDF / thesis discussion of Bartolomeo.

The thesis states that at the end of the text Bartolomeo replaces the four names associated with a table by a four-syllable artificial name formed from their first syllables; example: Tripode + Pepo + Corvus + Vetula mancina -> TRI PE COR VE. It also states `four hundred words -> twenty codewords`. That numerical statement is preserved as a secondary-source claim and is NOT used as a parameter unless reconciled with the primary witness.

## Source firewall

Before any new Voynich comparison is opened, the following must be committed:
1. folio/image manifest and hashes;
2. diplomatic transcription or explicit unresolved marker for every relevant item on 7r–8v;
3. graphical-relation table for braces/columns on 8v;
4. source-derived rule ledger;
5. machine-readable attested codeword table;
6. source-only metrics and controls;
7. freeze manifest with file hashes.

Forbidden before that freeze:
- tuning a transcription to improve Voynich similarity;
- inferring missing syllables from expected Voynich morphology;
- treating higher-level braces as recursive codewords without textual evidence;
- creating unattested Cartesian recombinations;
- merging the `De numeris ficticiis` mechanism with the syllabic codeword mechanism into a hybrid generator;
- mapping Latin names, syllables, body locations, directions or numerical values onto EVA/STA glyphs;
- changing the comparison metrics after opening the Voynich target.

## Two source-faithful subcorpora

### B4-A — Syllabic codewords on f.8v

Unit of analysis: each immediate bracketed group that visibly links source names to a compact/artificial form.

For every group record:
- manuscript order;
- source names, preserving diplomatic spelling;
- normalized reading where defensible;
- source first syllables;
- written compound/codeword exactly as present;
- bracket coordinates/level;
- left/right page half and higher graphical parent;
- reading confidence per field.

No synthetic codewords are permitted in the primary analysis.

### B4-B — `De numeris ficticiis`, f.7v–8r

This is analysed independently from B4-A.

Record only explicitly attested mappings among:
- numerical magnitude/value;
- written sign or number form;
- bodily/spatial locus;
- left/right or other positional state;
- ordinal/grade terminology;
- any explicit composition or interpretation rule.

Primary test here is architectural/positional, not string similarity. A linear surface-string generator is forbidden unless the manuscript explicitly supplies one.

## Transcription confidence

Each atomic reading receives one of:
- A = clear in supplied image and palaeographically secure;
- B = probable, minor abbreviation/letter uncertainty;
- C = tentative; structurally useful only if result is unchanged when excluded;
- U = unresolved; no normalized value may be imputed.

Primary metrics use A+B only. A-only and A+B+C are preregistered robustness views. U is always excluded, never guessed.

## Graphical structure on f.8v

Every brace or linking mark is entered independently of semantic interpretation:
- `edge_id`
- source node(s)
- target node/label if visible
- nesting level measured graphically
- side/region
- confidence

A higher-level brace may be described as `GROUPING` but not `RECURSIVE_ENCODING` unless independent textual evidence licenses that interpretation.

## Source-only metrics

### B4-A surface metrics
1. number of attested groups/codewords;
2. codeword character/syllable length;
3. component-position inventories;
4. H(component | slot) and slot marginal entropy;
5. pairwise slot mutual information;
6. total correlation where sample size permits;
7. exact edit-distance-1 graph: pair count, degree, isolated fraction;
8. edit-location distribution prefix/internal/suffix;
9. first-order character H(next|prev);
10. positional character entropy from left and from right;
11. right-minus-left positional entropy asymmetry;
12. final-component entropy versus initial-component entropy;
13. repeated component/sub-string reuse;
14. observed inventory / Cartesian capacity, reported descriptively only because prepared-codebook sampling is not generative.

### B4-B architecture metrics
1. number of independent dimensions explicitly required to decode a value;
2. whether position changes interpretation;
3. whether left/right is semantic;
4. whether the same sign/state is reused across positions;
5. hierarchy depth / magnitude grades;
6. compositionality versus prepared lookup;
7. ambiguity if position is removed;
8. information gain contributed by position where computable.

## Source-only controls

For B4-A:
- C0: slot-marginal shuffle of the attested four components, preserving each slot inventory and codeword count;
- C1: order-shuffle within each codeword, preserving its four components;
- C2: syllable-boundary-preserving character shuffle within components;
- C3: matched-length iid character strings using the attested character marginal.

10,000 deterministic replicates per stochastic control; seed namespace `VSN_B4_V1`.

These controls answer whether observed transition/right-edge structure is a property of Bartolomeo's prepared inventory rather than a trivial consequence of four concatenated syllables.

## Voynich comparison gate

Voynich target comparison remains CLOSED until the source freeze is complete.

After source freeze, compare the immutable historical fingerprint with the already-defined RF target, then STA-family/full-STA/AAA robustness representations. No parameter fitting.

Primary cross-system dimensions:
- H(next|prev);
- right-minus-left positional entropy;
- initial/final entropy asymmetry;
- edit-1 density and edit-location mixture;
- repeated-component reuse;
- length distribution.

The positional `De numeris ficticiis` system is compared only at mechanism/architecture level unless a source-attested string serialization exists.

## Decision rule

B4-A earns **SURFACE STRUCTURAL TRANSFER** only if, before looking at Voynich, the attested inventory independently has nontrivial constraint relative to its controls, and after unlock it matches the direction of BOTH:
1. Voynich transition constraint; and
2. Voynich right-edge asymmetry;
while not catastrophically failing edit-density/length structure.

B4-B earns **POSITIONAL MECHANISM TRANSFER** if the source establishes that interpretation depends materially on position/state in a way genuinely analogous at the architectural level. This cannot be promoted to a surface-string match.

Otherwise report PARTIAL TRANSFER, MECHANISM ONLY, or MISMATCH. No post-hoc hybrid rescue.

## Execution stages

0. Freeze protocol and source ledger.
1. Primary-source transcription 7r–8v and graphical graph extraction.
2. Independent secondary-source reconciliation; document disagreements without silently resolving them.
3. Freeze machine-readable B4-A and B4-B corpora.
4. Run source-only metrics and 10k controls.
5. Freeze source fingerprint + hashes.
6. Only if stages 0–5 complete, unlock immutable Voynich comparison.
7. Close B4 with a binding verdict.
