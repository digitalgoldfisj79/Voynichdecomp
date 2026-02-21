# A Formal Grammar of the Voynich Manuscript

A complete 4-slot morphological decomposition of every token in the Voynich Manuscript (Beinecke MS 408), with zero reconstruction errors across 37,465 tokens and 7,598 types.

## The claim

Every word in the Voynich Manuscript can be decomposed as:

```
w = prefix · gallows · core · suffix
```

using 8 prefixes, 9 gallows, an open set of ~2,000 cores, and 33 suffixes grouped into 7 families. The decomposition is lossless — concatenating the four slots reconstructs the original token in every case.

The chain-rule entropy budget sums exactly:

| Slot | H (bits) | % of total |
|------|----------|------------|
| H(prefix) | 2.788 | 27.0% |
| H(gallows \| prefix) | 1.374 | 13.3% |
| H(core \| prefix, gallows) | 3.622 | 35.1% |
| H(suffix \| prefix, gallows, core) | 2.527 | 24.5% |
| **Total** | **10.311** | **100.0%** |

Residual: 0.000 bits. No information is lost or unaccounted for.

## Verify it yourself

```bash
pip install numpy scipy
python p70_grammar_validation.py
```

This takes under 60 seconds and reproduces every number above from the data files. No API keys, no external dependencies beyond NumPy/SciPy.

The validation script also runs 19 alternative decompositions (random splits, fixed-position cuts, shifted boundaries, ablated slot inventories) and shows that none comes within 1 bit of the grammar's entropy budget. The nearest alternative is 1,074× further from the target.

## What's in this repo

| File | Description |
|------|-------------|
| `enriched_records.json` | 37,465 token decompositions with metadata header (self-documenting) |
| `enriched_records.pkl` | Same data as Python pickle (smaller, faster) |
| `p70_rules_canonical.json` | 210 segmentation rules (109 boundary-active, 101 coverage) with section-conditioned weights |
| `voynich_section_map.json` | Page-to-section mapping for all 9 canonical sections |
| `VMS_formal_grammar.pdf` | 2-page formal specification of the complete grammar |
| `p70_grammar_validation.py` | Validation script: reproduces all metrics and tests 19 alternatives |

The transcription source is the ZLZI (Zandbergen-Landini) transliteration from the [voynich.science](https://github.com/OrcusLabs/voynich.science) corpus in EVA (Extended Voynich Alphabet).

## The slot inventories

**Prefixes** (8): ∅, o, y, d, s, ch, sh, qo

**Gallows** (9): ∅, k, t, p, f, ckh, cth, cph, cfh

**Suffix families** (7 families, 33 types):

| Family | Members | Coverage |
|--------|---------|----------|
| Y | y, edy, ey, eey, eedy, dy, ody, chy, shy | 40.0% |
| N | aiin, ain, iin, n, aiiin, iiin, oiin, oiiin | 15.8% |
| L | ol, al, l | 15.6% |
| R | ar, or, r, ir | 14.9% |
| BARE | ∅ | 10.6% |
| M | am, m | 2.6% |
| OTHER | g, he, ee, b, ai, a, e, s | 0.5% |

**Core**: 2,001 types (open set, 1–10 characters). 52.7% of tokens have empty core — over half the manuscript consists of purely combinatorial prefix + (gallows) + suffix sequences with no core content.

## Key structural finding

The segmentation rules identified `ch` and `sh` as **prefixes**, not gallows-initial sequences. This is the single most consequential boundary decision. It affects 9,134 tokens (24.4%) directly, and cascades to change the parse of 24,964 tokens (66.6%) because it alters where suffix boundaries fall.

Example — the word `chedy` (503 occurrences):

| Parse | Prefix | Gallows | Core | Suffix |
|-------|--------|---------|------|--------|
| This grammar | ch | ∅ | ∅ | edy |
| Conventional | ∅ | ch | e | dy |

The conventional parse fragments the suffix (`edy` → `e` + `dy`) and creates a spurious core (`e`). The grammar's parse keeps `edy` as a single suffix unit and leaves the core empty — consistent with the 52.7% empty-core rate across the manuscript.

## Cross-slot coupling

The slots are not independent:

| Coupling | Value | Interpretation |
|----------|-------|----------------|
| Cramér's V (prefix × gallows) | 0.266 | Moderate co-selection |
| MI(suffix ; core) | 0.976 bits | Core-final character predicts suffix |
| MI(section ; prefix) | 0.067 bits | Prefix encodes positional/section info |
| MI(section ; core) | 0.348 bits | Section explains 8% of core entropy |

## What this grammar does not do

It does not decipher the manuscript. The core slot contains 2,001 opaque types carrying 35.1% of all token information. Without an external key, these remain uninterpretable. The suffix families have distributional roles characterised positionally but not semantically.

The grammar characterises the manuscript as a **structured notation system** — not a natural language (rigid 4-slot structure, mid-word entropy peak, Bernoulli empty-core/full-core alternation), not a simple cipher (section-specific vocabulary, folio coherence, label uniqueness), and not meaningless (genuine information discontinuity at word boundaries, page-specific content).

## Methodology

The segmentation rules were derived computationally from character-level statistics using iterative boundary detection and validated against entropy-based unsupervised segmentation (93% agreement within ±1 character). The rules were refined across 70+ iterations (designated p1–p70) with systematic falsification of alternatives at each stage.

Computational analysis was assisted by Claude (Anthropic). All results are deterministically reproducible from the published data and code.

## Data format

The dataset is available in two formats:

- `enriched_records.json` (12.5 MB) — self-documenting, includes metadata header with field descriptions and slot inventories
- `enriched_records.pkl` — Python pickle, smaller and faster to load

Both contain the same 37,465 records. Each record is a dictionary:

```python
{
    'token': 'chedy',        # Original EVA token
    'prefix': 'ch',          # ∅ if empty
    'gallows': '∅',          # ∅ if empty
    'core': '∅',             # ∅ if empty
    'suffix': 'edy',         # ∅ if empty
    'sfx_fam': 'Y',          # Suffix family
    'm_core': '∅',           # Modified core (internal)
    'empty_core': True,      # Boolean
    'section': 'Herbal-A',   # One of 9 canonical sections
    'folio': 'f2r',          # Folio identifier
    'line_no': 3,            # Line number within folio
    'pos': 2,                # Word position within line
    'line_len': 8,           # Total words in line
    'rel_pos': 0.286,        # Normalised position (0–1)
    'rel_line': 0.143,       # Normalised line position
    'is_first_word': False,
    'is_last_word': False,
    'is_first_line': False,
    'is_last_line': False
}
```

Load with:

```python
# JSON (any language)
import json
with open('enriched_records.json') as f:
    data = json.load(f)
records = data['records']          # 37,465 token decompositions
print(data['statistics'])          # corpus summary
print(data['slot_inventories'])    # prefix/gallows/suffix lists

# Pickle (Python only, faster)
import pickle
with open('enriched_records.pkl', 'rb') as f:
    records = pickle.load(f)
```

```javascript
// JavaScript / Node.js
const data = JSON.parse(require('fs').readFileSync('enriched_records.json'));
console.log(data.statistics);      // corpus summary
console.log(data.records[0]);      // first token
```

Note: the pickle file may have extension `.pkl.txt` depending on how it was exported. Both work with `pickle.load()`.

## The 9 canonical sections

| Section | Description |
|---------|-------------|
| Herbal-A | Herbal illustrations, Currier language A (Quires 1–8) |
| Herbal-B | Herbal illustrations, Currier language B (Quires 15, 17) |
| Astronomical | Astronomical diagrams |
| Cosmological | Cosmological diagrams |
| Zodiac | Zodiac pages |
| Rosettes | Rosettes foldout |
| Balneological | Bathing/biological figures |
| Pharmaceutical | Pharmaceutical/recipe pages |
| Stars | Star-labelled pages |

## Citation

If you use this decomposition in your own work, please reference this repository and the formal grammar specification.

## Licence

Data and code released under MIT licence. The Voynich Manuscript itself (Beinecke MS 408) is in the public domain.
