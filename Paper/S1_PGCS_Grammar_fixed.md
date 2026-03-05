S1: PGCS Grammar — Derivation, Specification, and Validation

Source code: src/p70c_full.py (P70C_Full class, builder)
Data: data/p70_rules_canonical.json (210 rules), data/p70c_full_spec_v1.json (6,750 quints), data/enriched_records.pkl (37,465 parsed tokens)

Terminology note. The implementation uses internal names that appear in source code headers throughout these supplements:

S1.1 Derivation Methodology

S1.1.1 Prior Art and Starting Point

The PGCS decomposition extends Stolfi’s (2005) crust–mantle–core grammar, which established a nested three-layer word structure with positional character constraints and 94.4% token acceptance. PGCS departs from Stolfi’s model in two ways: it separates gallows from core as a distinct functional slot (motivated by the cardinality difference: 9 gallows types vs. 2,001 core types), and it reclassifies the bench characters ch and sh from mantle to prefix status. A mapping between Stolfi’s categories and PGCS slots is given in §2.2 of the main paper.

S1.1.2 Rule Discovery

The 109 foundational rules (designated P69) were extracted computationally from character bigram co-occurrence patterns across the ZLZI corpus. For each character pair and word position, the algorithm computed the conditional probability of the pair appearing at a morphological boundary versus within a morphological unit, weighted by section-specific frequencies. Rules were retained when they exceeded a stability threshold across multiple manuscript sections.

Critically, this process discovered the prefix status of ch and sh. The boundary rules found that these characters cluster distributionally with word-initial characters (o, d, y, s, qo), not with gallows characters (k, t, p, f). This reclassification was not assumed; it emerged from the co-occurrence statistics and was subsequently confirmed by the information-theoretic evidence documented in §2.2 of the main paper and the 24-alternative comparison in the main README.

At P69, the system reached saturation: additional rules re-expressed existing structural bias without capturing new information. P69 was frozen as the foundational boundary model.

S1.1.3 Train/Test Protocol

After freezing P69, 40% of the corpus was held out by folio. This held-out set was not used during any subsequent stage of rule development or coverage extension. Results of the held-out test (3,930 unseen types, zero violations) and 50-trial cross-validation are reported in S4.1.

S1.1.4 Coverage Extension (P70)

Two further rounds extended rule coverage without altering P69’s boundary decisions:

The 101 additional rules address suffix patterns, gallows contexts, word-final/initial constraints, and compound suffixes. All follow the same schema with section-conditioned weights (documented in S1.3). The P69 boundary positions — including the ch/sh prefix classification — remained unchanged.

The iteration history is recorded in the created_from header of p70_rules_canonical.json. The full 210-rule set and 24-alternative comparison are reproducible via p70_grammar_validation.py (under 60 seconds, NumPy and SciPy only).

S1.2 Slot Inventories

Prefix Slot (8 types + ∅)

The reclassification of ch and sh from gallows to prefix status is the most consequential boundary decision in the PGCS model, affecting 11,329 tokens (30.2%). These characters behave distributionally as prefixes: they appear at word-initial position, combine freely with following gallows characters, and affect 9,134 tokens directly (24.4%); the cascading effect on suffix boundaries means that 24,964 tokens (66.6%) parse differently under PGCS than under any three-slot alternative.

Gallows Slot (9 types + ∅)

Core Slot (open class: 2,001 types + ∅)

The core is an open-class slot with 2,001 distinct values plus the empty core. Critically, 52.7% of all tokens (19,730) have empty cores, meaning they are composed entirely from closed-class inventory items (prefix + gallows + suffix).

Core length distribution: 0 characters (empty) 52.7%, 1 character 12.3%, 2 characters 18.1%, 3 characters 10.4%, 4+ characters 6.5%.

Suffix Slot (33 types, 7 families)

The suffix-family abstraction reduces 33 full suffixes to 7 families. The vowel-prefix component of each suffix carries 45% of full suffix entropy and 85% of section-specific information, while the terminal carries structural information. The family level captures grammar; the discarded variation is content.

S1.3 Character Grammar (210 Rules)

The character grammar (P70) comprises 210 rules in four categories:

Validation statistics: character coverage 92.96%, full parse rate 71.93% of word types, partial coverage 99.87%, held-out generalisation 0/3,930 novel types produce violations, decomposition error 0.001 bits (vs 1.074 bits for next-best alternative, roughly 1,000× improvement).

S1.4 Constrained Quad Inventory

The constrained quad inventory (PGCS-C) records the 5,172 observed quads with frequency and tier metadata.

From a theoretical space of 656,208 possible combinations at the classified level (8 prefixes × 9 gallows × 1,302 core-classes × 7 suffix families), or 1,008,504 at the raw level (using all 2,001 distinct core types), only 5,172 (0.79%) are observed — a 127× compression.

Tier distribution:

T1+T2 (435 entries, 77.6% of tokens) constitute the generalisable grammar. T4 entries are the observed lexicon: hapax quads that represent individual content words.

S1.5 Position-Conditioned Quint Layer

Adding line position (FIRST/MID/LAST) as a fifth axis produces 6,750 observed quintuples from a theoretical space of 1,968,624 — a 292× compression.

Position conditioning eliminates 58.6% of spurious (token, position) assignments:

This is placement precision (constraining where tokens appear), not vocabulary precision (which tokens exist). The vocabulary is unchanged by position conditioning.

S1.6 Ledger Compactness

The PGCS grammar can be expressed as a compact reference table — a “ledger” — small enough to fit on a single manuscript page. This section derives the component count cited in §2.4 of the main paper.

Component Inventory

The 231 core bigrams represent all attested two-character sequences within the core slot. These encode the character grammar’s core-internal constraints: which characters can follow which within the open-class slot. Combined with the prefix, gallows, and suffix inventories, they define the complete set of character-level legality constraints equivalent to the 210 rules documented in S1.3.

Page Capacity Comparison

A single densely-written page can accommodate the full grammar specification. The “approximately 450 table entries” figure cited in the main paper rounds up from 443 to account for formatting overhead that a practical reference table would require.

What the Ledger Does Not Contain

The ledger encodes grammar (the 28.9% structural layer) but not content (the 71.1% lexical layer). It does not contain the 2,001 distinct core morphemes, the section-specific vocabulary profiles, or the frequency weights that distinguish one passage from another. A scribe using only this ledger would produce grammatically valid text — satisfying the PGCS slot architecture, character constraints, and sequential transitions — but would need an additional source of lexical selection to reproduce the manuscript’s content signal.

Character-Level Over-Generation

The character adjacency constraints in the grammar (627 legal character bigram transitions from the VMS alphabet of 25 characters) define the space of all character strings the ledger can produce. Exhaustive enumeration of legal character walks at lengths 1–6 yields approximately 3.88 million distinct forms. Extrapolation to lengths 7–10 (the observed VMS word-length range) gives a conservative lower bound of 5.3 million legal token forms. Against the manuscript’s 7,598 attested types, this represents an over-generation ratio of at least 700×. The character-level ledger is thus dramatically under-constrained: it accepts any string obeying local bigram rules, regardless of whether the string constitutes a valid PGCS word. The slot grammar, by contrast, compresses the free combinatorial product 127-fold (S1.4), and attested quadruples compress it further still. This quantifies the empirical bite of the grammar relative to character-level description.

This separation is the physical analogue of the information budget (S2): the grammar fits on a leaf; the content does not.