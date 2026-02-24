# S1: PGCS Grammar Specification

> **Source code:** `src/p70c_full.py` (P70C_Full class, builder)  
> **Data:** `data/p70_rules_canonical.json` (210 rules), `data/p70c_full_spec_v1.json` (6,750 quints)

**Terminology note.** The implementation uses internal names that appear in source code headers throughout these supplements. Their correspondence to the main paper's terminology:

| Internal name | Main paper term | Description |
|---------------|----------------|-------------|
| P70 | Character grammar (210 rules) | Character-level bigram legality constraints validating PGCS slot boundaries |
| P70-C / PGCS-C | Constrained quad inventory (5,172 quads) | Attested slot-filling combinations with frequency and tier metadata |
| `enriched_records.pkl` | Enriched corpus (37,465 tokens) | Full P70-C parse applied to every token, with section, folio, and position metadata |
| Quad | Quad | Four-slot combination: (prefix, gallows, core-class, suffix-family) |
| Quint | Quint | Quad plus line position (FIRST/MID/LAST) |


## S1.1 Slot Inventories

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

The reclassification of *ch* and *sh* from gallows to prefix status is the most consequential boundary decision in the PGCS model, affecting 11,329 tokens (30.2%). These characters behave distributionally as prefixes: they appear at word-initial position, combine freely with following gallows characters, and account for 66.6% of all tokens containing them (24,964 tokens).

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

Core length distribution:
- 0 characters (empty): 52.7%
- 1 character: 12.3%
- 2 characters: 18.1%
- 3 characters: 10.4%
- 4+ characters: 6.5%

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

## S1.2 Character Grammar (210 Rules)

The character grammar (P70) comprises 210 rules in four categories:

| Category | Rules | Coverage |
|----------|-------|----------|
| Character-sequence constraints | 81 | Bigram legality |
| Pair-adjacency rules | 52 | Cross-boundary constraints |
| Suffix rules | 41 | Suffix formation |
| Prefix rules | 36 | Prefix combinatorics |

### Validation Statistics

- Character coverage: 92.96%
- Full parse rate: 71.93% of word types
- Partial coverage: 99.87%
- Held-out generalisation: 0/3,930 novel types produce violations
- Decomposition error: 0.001 bits (vs 1.074 bits for next-best alternative; ~1,000× improvement)

## S1.3 Constrained Quad Inventory

The constrained quad inventory (PGCS-C) records the 5,172 observed quads with frequency and tier metadata.

From a theoretical space of 656,208 possible combinations (8 prefixes × 9 gallows × 2,001 cores × 7 suffix families), only 5,172 (0.79%) are observed — a 127× compression.

**Tier distribution:**

| Tier | n range | Quads | Token coverage |
|------|---------|-------|---------------|
| T1 | ≥50 | 118 | 61.3% |
| T2 | 10–49 | 317 | 16.3% |
| T3 | 4–9 | 511 | 7.8% |
| T4 | 1–3 | 4,226 | 14.6% |

T1+T2 (435 entries, 77.6% of tokens) constitute the generalisable grammar. T4 entries are the observed lexicon: hapax quads that represent individual content words.

## S1.4 Position-Conditioned Quint Layer

Adding line position (FIRST/MID/LAST) as a fifth axis produces 6,750 observed quintuples from a theoretical space of 1,968,624 — a 292× compression.

Position conditioning eliminates 58.6% of spurious (token, position) assignments:

| Position | Valid quads | Eliminated | Reduction |
|----------|-----------|------------|-----------|
| FIRST | 1,830 | 3,342 | 64.6% |
| MID | 3,458 | 1,714 | 33.1% |
| LAST | 1,462 | 3,710 | 71.7% |

This is placement precision (constraining where tokens appear), not vocabulary precision (which tokens exist). The vocabulary is unchanged by position conditioning.
