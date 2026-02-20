# v3d Positional Encoding vs P70 Morphological Fingerprint

## Input Data
- **VMS**: 34,416 tokens / 7,178 types / 217 pages (ZLZI transcription)
- **Latin**: 8,679 tokens / 2,975 types / 77 recipes (Antidotarium Nicolai, cleaned)
- **v3d output**: 8,550 tokens / 3,077 types / 77 encoded recipes
- **P70**: 210 rules (109 boundary-active) across 9 canonical sections

---

## TEST A: Surface Metrics

| Metric | v3d | VMS | Match% |
|---|---|---|---|
| Tokens | 8,550 | 34,416 | 24.8% (corpus size mismatch) |
| Types | 3,077 | 7,178 | 42.9% |
| Type/Token Ratio | 0.360 | 0.209 | 27.4% |
| Mean word length | 5.73 | 4.99 | 85.3% ✓ |
| Hapax ratio | 0.650 | 0.683 | 95.2% ✓ |
| Char entropy | 3.71 | 3.86 | 96.2% ✓ |
| Word entropy | 9.99 | 10.29 | 97.1% ✓ |
| Alphabet size | 18 | 26 | 69.2% |

**Average surface match: 67.3%** (73.5% excluding corpus-size metrics N/V/TTR)

### Word Length Distribution
| Length | v3d | VMS | Delta |
|---|---|---|---|
| 2 | 3.7% | 6.2% | -2.5 |
| 3 | 8.6% | 9.2% | -0.6 |
| 4 | 13.2% | 17.7% | -4.4 |
| 5 | 26.1% | 24.8% | +1.3 |
| 6 | 15.3% | 19.6% | -4.3 |
| 7 | 13.9% | 11.6% | +2.3 |
| 8+ | 19.0% | 7.3% | +11.7 ✗ |

v3d overproduces long words (8+ chars: 19% vs 7.3%). The real VMS tends shorter.

---

## TEST B: P70 Slot Decomposition

| Metric | v3d | VMS | Assessment |
|---|---|---|---|
| Any prefix or suffix | 99.0% | 95.7% | Close |
| Both prefix AND suffix | 64.5% | 51.3% | v3d over-slotted (+13%) |
| Unique cores | 1,110 | 2,084 | v3d under-diverse |
| Mean core length | 3.6 | 3.4 | Close |

### Prefix Distribution
| Prefix | v3d | VMS | Delta | Issue |
|---|---|---|---|---|
| (none) | 32.5% | 41.7% | -9.2 | v3d under-represents null prefix |
| 'd' | 14.8% | 7.7% | +7.1 | **v3d double-maps (d+m→d)** |
| 'o' | 26.9% | 22.3% | +4.5 | Slight excess |
| 'q' | 0.0% | 1.2% | -1.2 | **Missing entirely** |
| 'qo' | 9.5% | 9.2% | +0.3 | Good match |
| 's' | 16.3% | 10.5% | +5.8 | **v3d over-maps (f→s)** |
| 'y' | 0.0% | 7.3% | -7.3 | **Missing entirely** |

### Key Suffix Divergences
| Suffix | v3d | VMS | Issue |
|---|---|---|---|
| (none) | 4.0% | 11.2% | **v3d under-produces bare forms** |
| 'edy' | 10.3% | 5.1% | v3d over-produces (too many -as/-ut/-rum maps) |
| 'ain' | 7.3% | 3.3% | v3d over-produces (too many -em/-at/-nt maps) |
| 'eey' | 6.6% | 3.3% | v3d over-produces (too many -es maps) |
| 's' | 0.0% | 6.0% | **Missing entirely** |
| 'chy' | 0.4% | 2.8% | Under-represented |
| 'shy' | 0.0% | present | **Missing** |

---

## TEST C: Prefix × Suffix Co-occurrence

| Metric | v3d | VMS |
|---|---|---|
| **Cramér's V** | **0.332** | **0.211** |
| N (tokens with both slots) | 5,360 | 17,725 |

v3d's prefix-suffix coupling is **57% stronger** than VMS. The real system has more independence between initial and final slots.

### Attraction/Repulsion Agreement (direction match)
| VMS pair | VMS R | v3d R | Direction match? |
|---|---|---|---|
| d...iin | +31.5 | +4.1 | ✓ (but much weaker) |
| qo...dy | +12.0 | +5.5 | ✓ |
| d...r | +12.0 | +13.8 | ✓ |
| s...ol | +11.7 | +7.1 | ✓ |
| d...ey | -12.0 | -5.5 | ✓ (but weaker) |
| d...edy | -11.3 | -8.5 | ✓ |
| s...iin | -10.6 | -0.8 | ✓ (barely) |

v3d reproduces the **direction** of all VMS co-occurrence patterns, but with weaker magnitudes on attractions and an overall too-tight coupling.

---

## TEST D: Slot Inventory Overlap

### Prefixes (freq ≥ 3)
- v3d: 4 prefixes {d, o, qo, s}
- VMS: 6 prefixes {d, o, q, qo, s, **y**}
- Missing: **q** (bare, without -o), **y**

### Suffixes (freq ≥ 3)
- v3d: 22 suffixes
- VMS: 28 suffixes
- Overlap: 22 (79% of VMS inventory)
- Missing: an, g, h (rare), n, **s**, **shy**

### Cores (freq ≥ 2)
- v3d: 475 cores
- VMS: 636 cores
- **Overlap: 96 (20.2% of v3d, 15.1% of VMS)**

The core overlap is **catastrophically low**. v3d's hash-enumerated medial encoding produces a fundamentally different core vocabulary than the real VMS.

---

## TEST E: Gallows Behaviour

| Metric | v3d | VMS |
|---|---|---|
| Gallows-initial overall | 4.5% | 7.6% |
| Gallows at para start | 74.0% | 69.6% |
| 't' rate | 1.9% | 2.6% |
| 'k' rate | 1.7% | 3.3% |
| 'p' rate | 1.0% | 1.4% |
| 'f' rate | 0.0% | 0.3% |

v3d under-produces gallows overall (4.5% vs 7.6%) but nails the paragraph-initial gallows rate (74% vs 70%). The logogram-trigger mechanism works for paragraph starts but misses non-initial gallows occurrences — suggesting the real system embeds gallows via a second mechanism beyond template absorption.

---

## SYNTHESIS: What Must Change for v4

### 1. Missing Prefix Sources (CRITICAL)
The real system produces 'y' prefix (7.3% of types) and bare 'q' prefix (1.2%). These have no source in v3d's Latin initial-consonant mapping. Possible explanations:
- 'y' encodes a grammatical category (not phonographic)
- 'q' without 'o' may come from a different source layer

### 2. Core Encoding Must Be Structural, Not Hash-Based (CRITICAL)
Only 15% core overlap. The real VMS reuses a constrained set of medial forms (636 with freq≥2) built from bench characters and connectors. v3d's enumeration-based assignment produces the wrong vocabulary. **The medial slot must be deterministic from the source word's structure, not arbitrary.**

### 3. Suffix Distribution Needs Rebalancing
v3d over-produces complex suffixes (edy, ain, eey) and under-produces bare forms (null, dy). The real system appears to truncate or simplify suffixes more aggressively.

### 4. Prefix-Suffix Decorrelation
Cramér's V 0.332 vs 0.211. Something in the real system partially decorrelates prefix from suffix — possibly position-dependent variation operating more on the suffix than v3d's family rotation.

### 5. Gallows Need a Second Source
Beyond template logograms, the real VMS inserts gallows within running text (raising overall rate from 4.5% to 7.6%). This could be: abbreviation markers, special consonant clusters, or a numeral/quantity encoding layer.

### 6. Alphabet Coverage
v3d uses 18 characters; VMS uses 26. The 8 missing characters likely encode distinctions v3d collapses.
