# S1: PGCS Grammar — Derivation, Specification, and Validation

> **Source code:** `src/p70c_full.py` (P70C_Full class, builder)  
> **Data:** `data/p70_rules_canonical.json` (210 rules), `data/p70c_full_spec_v1.json` (6,750 quints), `data/enriched_records.pkl` (37,465 parsed tokens)

**Terminology note.** The implementation uses internal names that appear in source code headers throughout these supplements:

| Internal name | Main paper term | Description |
|---------------|----------------|-------------|
| P70 | Character grammar (210 rules) | Character-level bigram legality constraints validating PGCS slot boundaries |
| P70-C / PGCS-C | Constrained quad inventory (5,172 quads) | Attested slot-filling combinations with frequency and tier metadata |
| `enriched_records.pkl` | Enriched corpus (37,465 tokens) | Full P70-C parse applied to every token, with section, folio, and position metadata |
| Quad | Quad | Four-slot combination: (prefix, gallows, core-class, suffix-family) |
| Quint | Quint | Quad plus line position (FIRST/MID/LAST) |


## S1.1 Derivation Methodology

### S1.1.1 Prior Art and Starting Point

The PGCS decomposition extends Stolfi's (2005) crust–mantle–core grammar, which established a nested three-layer word structure with positional character constraints and 94.4% token acceptance. PGCS departs from Stolfi's model in two ways: it separates gallows from core as a distinct functional slot (motivated by the cardinality difference: 9 gallows types vs. 2,001 core types), and it reclassifies the bench characters *ch* and *sh* from mantle to prefix status. A mapping between Stolfi's categories and PGCS slots is given in §2.2 of the main paper.

### S1.1.2 Rule Discovery

The 109 foundational rules (designated P69) were extracted computationally from character bigram co-occurrence patterns across the ZLZI corpus. For each character pair and word position, the algorithm computed the conditional probability of the pair appearing at a morphological boundary versus within a morphological unit, weighted by section-specific frequencies. Rules were retained when they exceeded a stability threshold across multiple manuscript sections.

Critically, this process *discovered* the prefix status of *ch* and *sh*. The boundary rules found that these characters cluster distributionally with word-initial characters (*o*, *d*, *y*, *s*, *qo*), not with gallows characters (*k*, *t*, *p*, *f*). This reclassification was not assumed; it emerged from the co-occurrence statistics and was subsequently confirmed by the information-theoretic evidence documented in §2.2 of the main paper and the 24-alternative comparison in the main README.

At P69, the system reached saturation: additional rules re-expressed existing structural bias without capturing new information. P69 was frozen as the foundational boundary model.

### S1.1.3 Train/Test Protocol

After freezing P69, 40% of the corpus was held out by folio. This held-out set was not used during any subsequent stage of rule development or coverage extension. Results of the held-out test (3,930 unseen types, zero violations) and 50-trial cross-validation are reported in S4.1.

### S1.1.4 Coverage Extension (P70)

Two further rounds extended rule coverage without altering P69's boundary decisions:

| Round | Rules added | Cumulative | Character coverage | Word full coverage |
|-------|------------|------------|-------------------|-------------------|
| P69 (frozen) | 109 | 109 | 45.61% | 1.30% |
| P70 round 1 | +71 | 180 | — | — |
| P70 round 2 | +30 | 210 | 92.96% | 71.93% |

The 101 additional rules address suffix patterns, gallows contexts, word-final/initial constraints, and compound suffixes. All follow the same schema with section-conditioned weights (documented in S1.3). The P69 boundary positions — including the ch/sh prefix classification — remained unchanged.

The iteration history is recorded in the `created_from` header of `p70_rules_canonical.json`. The full 210-rule set and 24-alternative comparison are reproducible via `p70_grammar_validation.py` (under 60 seconds, NumPy and SciPy only).


## S1.2 Slot Inventories

### Prefix Slot (8 types + ∅)

| Prefix | Frequency | % of corpus | Notes |
|--------|-----------|-------------|-------|
| ∅ | 11,842 | 31.6% | Empty prefix (default) |
| ch | 7,126 | 19.0% | Reclassified from gallows |
| qo | 5,489 | 14.7% | Always precedes gallows |
| sh | 4,203 | 11.2% | Reclassified from gallows |
| o | 3,891 | 10.4% | |
| d | 2,876 | 7.7% | Elevated at line boundaries |
| y | 1,127 | 3.0% | Line-initial marker |
| s | 911 | 2.4% | Line-initial marker |

The reclassification of *ch* and *sh* from gallows to prefix status is the most consequential boundary decision in the PGCS model, affecting 11,329 tokens (30.2%). These characters behave distributionally as prefixes: they appear at word-initial position, combine freely with following gallows characters, and affect 9,134 tokens directly (24.4%); the cascading effect on suffix boundaries means that 24,964 tokens (66.6%) parse differently under PGCS than under any three-slot alternative.

### Gallows Slot (9 types + ∅)

| Gallows | Frequency | % of corpus | Notes |
|---------|-----------|-------------|-------|
| ∅ | 18,247 | 48.7% | No gallows (empty core tokens) |
| k | 7,892 | 21.1% | Most frequent gallows |
| t | 5,641 | 15.1% | |
| p | 2,873 | 7.7% | Elevated at line-initial |
| f | 1,498 | 4.0% | |
| ckh | 589 | 1.6% | Bench variant |
| cth | 412 | 1.1% | Bench variant |
| cph | 187 | 0.5% | Bench variant |
| cfh | 98 | 0.3% | Bench variant |
| m | 128 | 0.3% | Rare |

### Core Slot (open class: 2,001 types + ∅)

The core is an open-class slot with 2,001 distinct values plus the empty core. Critically, 52.7% of all tokens (19,730) have empty cores, meaning they are composed entirely from closed-class inventory items (prefix + gallows + suffix).

Core length distribution: 0 characters (empty) 52.7%, 1 character 12.3%, 2 characters 18.1%, 3 characters 10.4%, 4+ characters 6.5%.

### Suffix Slot (33 types, 7 families)

| Family | Members | Frequency | % of corpus |
|--------|---------|-----------|-------------|
| Y | y, dy, ey, ody, eey, chy, oedy, edy, eedy, oey | 18,241 | 48.7% |
| N | aiin, ain, in, iin, oiin | 8,926 | 23.8% |
| L | ol, al, l, oal | 3,012 | 8.0% |
| R | ar, or, r | 2,567 | 6.9% |
| BARE | (empty) | 2,891 | 7.7% |
| M | am, om, m | 1,234 | 3.3% |
| S | os, as, es, s | 594 | 1.6% |

The suffix-family abstraction reduces 33 full suffixes to 7 families. The vowel-prefix component of each suffix carries 45% of full suffix entropy and 85% of section-specific information, while the terminal carries structural information. The family level captures grammar; the discarded variation is content.


## S1.3 Character Grammar (210 Rules)

The character grammar (P70) comprises 210 rules in four categories:

| Category | Rules | Coverage |
|----------|-------|----------|
| Character-sequence constraints | 81 | Bigram legality |
| Pair-adjacency rules | 52 | Cross-boundary constraints |
| Suffix rules | 41 | Suffix formation |
| Prefix rules | 36 | Prefix combinatorics |

Validation statistics: character coverage 92.96%, full parse rate 71.93% of word types, partial coverage 99.87%, held-out generalisation 0/3,930 novel types produce violations, decomposition error 0.001 bits (vs 1.074 bits for next-best alternative, roughly 1,000× improvement).


## S1.4 Constrained Quad Inventory

The constrained quad inventory (PGCS-C) records the 5,172 observed quads with frequency and tier metadata.

From a theoretical space of 656,208 possible combinations (8 prefixes × 9 gallows × 2,001 cores × 7 suffix families), only 5,172 (0.79%) are observed — a 127× compression.

Tier distribution:

| Tier | n range | Quads | Token coverage |
|------|---------|-------|---------------|
| T1 | ≥50 | 118 | 61.3% |
| T2 | 10–49 | 317 | 16.3% |
| T3 | 4–9 | 511 | 7.8% |
| T4 | 1–3 | 4,226 | 14.6% |

T1+T2 (435 entries, 77.6% of tokens) constitute the generalisable grammar. T4 entries are the observed lexicon: hapax quads that represent individual content words.


## S1.5 Position-Conditioned Quint Layer

Adding line position (FIRST/MID/LAST) as a fifth axis produces 6,750 observed quintuples from a theoretical space of 1,968,624 — a 292× compression.

Position conditioning eliminates 58.6% of spurious (token, position) assignments:

| Position | Valid quads | Eliminated | Reduction |
|----------|-----------|------------|-----------|
| FIRST | 1,830 | 3,342 | 64.6% |
| MID | 3,458 | 1,714 | 33.1% |
| LAST | 1,462 | 3,710 | 71.7% |

This is placement precision (constraining where tokens appear), not vocabulary precision (which tokens exist). The vocabulary is unchanged by position conditioning.


## S1.6 Ledger Compactness

The PGCS grammar can be expressed as a compact reference table — a "ledger" — small enough to fit on a single manuscript page. This section derives the component count cited in §4.4 of the main paper.

### Component Inventory

| Component | Entries | What it encodes |
|-----------|---------|----------------|
| Character alphabet | 25 | EVA glyph inventory |
| Prefix × Gallows pairs | 65 | Legal slot combinations (of 72 theoretical) |
| Suffix inventory (VP + terminal) | 66 | 33 suffixes decomposed into vowel-prefix and terminal components |
| Suffix-family transition weights | 56 | Non-zero cells in the 7-family × 8-prefix transition matrix |
| Core character bigrams | 231 | Legal character pairs within core slot |
| **Total** | **443** | |

The 231 core bigrams represent all attested two-character sequences within the core slot. These encode the character grammar's core-internal constraints: which characters can follow which within the open-class slot. Combined with the prefix, gallows, and suffix inventories, they define the complete set of character-level legality constraints equivalent to the 210 rules documented in S1.3.

### Page Capacity Comparison

| Measure | Value |
|---------|-------|
| Ledger entries | 443 |
| Median VMS page (characters) | 491 |
| Mean VMS page (characters) | 826 |
| Ledger as % of median page | ~90% |
| Ledger as % of mean page | ~54% |

A single densely-written page can accommodate the full grammar specification. The "approximately 450 table entries" figure cited in the main paper rounds up from 443 to account for formatting overhead that a practical reference table would require.

### What the Ledger Does Not Contain

The ledger encodes grammar (the 28.9% structural layer) but not content (the 71.1% lexical layer). It does not contain the 2,001 distinct core morphemes, the section-specific vocabulary profiles, or the frequency weights that distinguish one passage from another. A scribe using only this ledger would produce grammatically valid text — satisfying the PGCS slot architecture, character constraints, and sequential transitions — but would need an additional source of lexical selection to reproduce the manuscript's content signal.

This separation is the physical analogue of the information budget (S2): the grammar fits on a leaf; the content does not.

