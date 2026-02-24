# PGCS Constraint Hierarchy — Operational Guide

## The Stack

```
Layer 0:  enriched_records.pkl       THE DATA         37,465 tokens, fully decomposed
Layer 1:  p70_rules_canonical.json   CHARACTER        210 rules, char adjacency
Layer 2:  p70c_full_layer.pkl        SLOT+POSITION    6,750 quints + transitions
                                     +TRANSITIONS     + paragraph markers
                                     +METADATA        + quire/production units
                                     +FULL SUFFIX     + VP+terminal access
```

These are not alternatives. They are a hierarchy — each layer operates at a different abstraction level and constrains different things.

---

## What Each Layer Is

### enriched_records.pkl — The Ground Truth

Every VMS token decomposed into 4 slots plus metadata:

```python
{
  'token': 'qokeedy',
  'prefix': 'qo',       # 8 values: ∅, ch, d, o, qo, s, sh, y
  'gallows': 'k',        # 9 values: ∅, cph, ckh, cth, f, k, p, t, cfh
  'core': 'eed',         # raw core string
  'suffix': 'y',         # raw suffix string
  'sfx_fam': 'Y',        # 7 families: BARE, L, M, N, OTHER, R, Y
  'm_core': 'ee',        # minimal core (suffix-stripped)
  'section': 'Herbal-A', # 9 sections
  'folio': 'f1r',
  'line_no': 1, 'pos': 3,
  'is_first_word': False, 'is_last_word': False,
  ...
}
```

This is the authority. Every other layer derives from it. If something contradicts the pkl, the pkl wins.

**Use for:** any analysis, building new models, validating claims, extracting statistics.

### p70_rules_canonical.json — Character-Level Grammar

210 rules constraining character adjacency within and between tokens:

- 81 **chargram** rules: "bigram X tends to appear in these sections"
- 36 **prefix** rules: "these character sequences begin words"
- 41 **suffix** rules: "these character sequences end words"
- 52 **pair** rules: "these cross-word-boundary patterns occur"

Each rule has: pattern, prediction side (left/right), base weight, allowed sections.

**What it knows:** which characters can follow which characters. Phonotactics.
**What it doesn't know:** slot boundaries. 19/25 characters appear in multiple PGCS slots. The rule `ch→left` can't distinguish prefix-ch from core-ch.

**Use for:** validating character sequences, testing whether a string *could* be Voynichese at the character level, the p70 grammar validation paper.

### p70c_full_layer.pkl — Slot Co-occurrence + Position + Transitions + Metadata

6,750 observed (Prefix, Gallows, m_core, SfxFam, Position) quints from the enriched pkl, plus:

- **Transition grammar**: 8×8 lookup table of P(prefix | prev_sfx_fam). Adds 0.757 bits beyond position+section. Encoded as lookup, not axis, to avoid sparsity.
- **Paragraph markers**: Detection method for first-line-of-paragraph openers (∅-prefix, gallows-bearing, FC). Too sparse (226 tokens) for a constraint axis.
- **Quire metadata**: Folio→quire mapping and production cluster assignments (e.g., f42-f49-f56). Quire adds 0.324 bits beyond section but is largely absorbed by it.
- **Full suffix access**: Both sfx_fam (7 values, for grammar) and full suffix (33 values, for content analysis) are available per quad/quint.

**What it knows:** which slot *combinations* actually occur, at which line positions, following which suffix families, in which sections and quires. Morphotactics + sequential grammar.
**What it doesn't know:** internal character structure of multi-character m_cores (that's p70's territory).

**Use for:** generation, validation, filtering, scoring, sequential text production.

---

## How They Work Together

### For ANALYSIS (pulling VMS apart)

```
Token "qokeedy" at line position 3 of 8, following "chol" (sfx_fam=L)
  ↓
  enriched_records.pkl: P=qo, G=k, mc=ee, SF=Y, pos=MID, section=Herbal-A
  ↓
  p70c_full: is_valid('qo','k','ee','Y','MID') → True, tier=T2, prob=0.0023
             transition_prob('L', 'qo') → 0.076  (L→qo is weak)
             get_full_suffixes('qo','k','ee','Y') → ['dy','edy','eedy','ey','y']
  ↓
  p70 rules: 'qo' matches prefix rule, 'ke' matches chargram, 'y' matches suffix
  ↓
  All layers agree. Token is fully characterised + transition scored.
```

### For GENERATION (producing VMS-like text)

```
Step 1: Choose line parameters (n_words, section, is_first_line)
  ↓
Step 2: layer.generate_line(n_words, section='Herbal-A', first_line=False)
        Internally:
        a) Assigns position (FIRST/MID/LAST) per word
        b) First word: if first_line, samples paragraph opener
        c) Each word: sample_quint_sequential(pos, prev_sfx)
           - Reweights prefix by transition_lookup
        d) Picks attested token from quint
  ↓
Step 3: Returns list of (token, quint) pairs
  ↓
Step 4 (optional): p70 rules verify character transitions
```

Or manually:

```python
from p70c_full import build_p70c_full
layer = pickle.load(open('p70c_full_layer.pkl', 'rb'))

# Sequential generation with transition grammar
prev_sfx = 'LINE_START'
for i in range(8):
    pos = 'FIRST' if i == 0 else ('LAST' if i == 7 else 'MID')
    quint = layer.sample_quint_sequential(pos, prev_sfx, min_tier='T2')
    toks = layer.get_tokens(*quint[:4], pos=quint[4])
    tok = random.choice(toks)
    print(tok, end=' ')
    prev_sfx = quint[3]  # sfx_fam feeds into next word
```

### For VALIDATION (testing whether something is VMS-like)

```
Input: candidate token "qotchedy" claimed to be line-initial in Herbal-B,
       following a Y-suffix word

  p70c_full.is_valid('qo','t','ce','Y','FIRST')
    → True/False (does this quint exist?)

  p70c_full.tier('qo','t','ce','Y','FIRST')
    → T1/T2/T3/T4/None (how common is it?)

  p70c_full.probability('qo','t','ce','Y','FIRST', section='Herbal-B')
    → 0.00xx (how likely given section+position?)

  p70c_full.transition_prob('Y', 'qo')
    → 0.26 (is qo-prefix expected after Y-suffix? YES, strongly)
```

### For SCORING (comparing models or passages)

```python
# Score a line using full model (position + transitions)
result = layer.score_line(line_records, section='Herbal-A')
print(f"Perplexity: {result['perplexity']:.1f}")
print(f"Mean surprisal: {result['mean_bits']:.2f} bits/token")

# Per-token breakdown
for t in result['tokens']:
    print(f"  {t['token']:15s} {t['bits']:.2f} bits  "
          f"p_pos={t['p_pos']:.4f}  p_trans={t['p_trans']:.3f}")
```

Lower perplexity = more typical VMS. Use this to compare sections, folios, scribal hands, or to detect anomalous passages.

---

## The Hierarchy at a Glance

```
                         Abstraction
                             ↑
  p70c_full_layer.pkl   SLOT COMBINATIONS    "P=qo + G=k + SF=Y is valid at LAST"
       constrains              |              6,750 quints, 292× compression
       what COMBINATIONS       |              + transition grammar (8×8 lookup)
       of slots occur          |              + paragraph markers, quire metadata
       and WHERE/WHEN          |              + full suffix access
                               |
  p70_rules_canonical.json  CHARACTERS       "q→o is a valid prefix start"
       constrains              |              210 rules
       what CHARACTERS         |
       can be adjacent         |
                               |
  enriched_records.pkl      RAW DATA         37,465 tokens, fully decomposed
       the actual              |              0 exceptions, 0 ambiguity
       observations            |
                             ↓
                          Concreteness
```

P70 rules are BELOW p70c_full in the hierarchy. P70 constrains character sequences. P70c_full constrains slot combinations, position, transitions, and metadata. They are complementary:

- p70c_full says "P=qo, G=k, mc=∅, SF=Y is valid at LAST, after Y-suffix"
- p70 says "within that, the character sequence q-o-k-e-y has valid transitions"
- Neither alone is sufficient; together they capture ~95% of VMS structure

---

## Practical Recipes

### "Give me all valid tokens at line-end in Stars section"

```python
layer = pickle.load(open('p70c_full_layer.pkl', 'rb'))
last_tokens_stars = set()
for q in layer.valid_quints:
    if q[4] == 'LAST':
        toks = layer.get_tokens(*q[:4], pos='LAST')
        last_tokens_stars.update(toks)
# Filter further by section using enriched_records if needed
```

### "What m_cores can follow prefix=qo, gallows=k at line-start?"

```python
mcores = layer.get_allowed_mcores('qo', 'k', sf='Y', pos='FIRST')
```

### "Generate a paragraph-opening line for Herbal-A"

```python
line = layer.generate_line(10, section='Herbal-A', first_line=True, min_tier='T2')
print(' '.join(tok for tok, quint in line))
```

### "What prefix should follow a Y-suffix word?"

```python
dist = layer.transition_dist('Y')
# {'qo': 0.26, 'o': 0.198, '∅': 0.193, 'ch': 0.13, ...}
```

### "What full suffixes exist for this quad?"

```python
full = layer.get_full_suffixes('qo', 'k', 'e', 'Y')
# ['chy', 'dy', 'edy', 'eedy', 'eey', 'ey', 'ody', 'shy']
```

### "Which quire does f49v belong to, and is it in a production cluster?"

```python
print(layer.quire_for_folio('f49v'))        # 'Q7'
print(layer.production_cluster('f49v'))     # 'CLUSTER_A'
```

### "Score this passage relative to section baseline"

```python
result = layer.score_line(passage_records, section='Herbal-A')
print(f"Mean surprisal: {result['mean_bits']:.2f} bits/token")
```

---

## File Inventory

| File | What | Size | Format |
|------|------|------|--------|
| enriched_records.pkl | Ground truth decomposition | 3.8MB | pickle (list of dicts) |
| p70_rules_canonical.json | Character grammar | 165KB | JSON |
| **p70c_full.py** | **Complete module (PRIMARY)** | ~18KB | Python (importable) |
| **p70c_full_layer.pkl** | **Pre-built full layer (PRIMARY)** | ~3MB | pickle (P70C_Full) |
| **p70c_full_spec_v1.json** | **Quint table + transitions** | ~2MB | JSON |
| p70c_pos_builder.py | Position-only module (superseded) | ~12KB | Python |
| p70c_pos_layer.pkl | Position-only layer (superseded) | ~2MB | pickle |
| transition_lookup.json | Standalone transition matrix (embedded in full) | 2KB | JSON |
| p70c_builder.py | Quad-only builder (superseded) | ~10KB | Python |
| p70c_layer.pkl | Quad-only layer (superseded) | ~1MB | pickle |
| p70c_connect.py | Connection analysis script | ~12KB | Python |
| p70c_connections.pkl | Connection results | ~50KB | pickle |

The pos-conditioned versions supersede the quad-only versions. Use p70c_pos_* for all new work. The quad-only files are kept for backwards compatibility and for cases where position is unknown or irrelevant.

---

## Connected Prior Findings

All four previously-explored phenomena have been tested against p70-c.

### Word-to-word transitions → LOOKUP TABLE

prev_sfx adds 0.757 bits beyond position+section — the single largest additional gain. But encoding as a 6th axis inflates to 9,951 sextuples with worse generalisation (19.2% miss vs 13.0%).

**Solution:** `transition_lookup.json` — an 8×8 matrix of P(prefix | prev_sfx). Use for sequential generation by reweighting p70c_pos prefix probabilities:

```python
import json
with open('transition_lookup.json') as f:
    trans = json.load(f)

# After sampling a quint from p70c_pos:
prev_sf = previous_word['sfx_fam']  # or 'LINE_START'
prefix_probs = trans[prev_sf]       # {'ch': 0.13, 'qo': 0.26, ...}
# Reweight the sampled quint or resample with prefix bias
```

Key rules: Y→qo (26%, 1.9×), BARE→∅ (39.5%, 1.8×), R→o/∅ (26%/26%), LINE_START→y/s/d enriched.

### Line-level patterns → METADATA

Paragraph markers (0.030 bits beyond pos+sec) are too sparse for a constraint axis. The three-zone grammar (opener/middle/closer) is already captured by the FIRST/MID/LAST position axis. Record `is_first_line` as metadata on the line, not as a p70-c dimension.

### Quire structure → METADATA

Quire adds 0.518 bits beyond section — real but finer-grained than section (16 quires vs 9 sections). It captures within-section variation (e.g., different herbal sub-batches, the f42-f49-f56 production cluster). Record as folio-level metadata, not a token-level constraint.

### Suffix decomposition → sfx_fam IS CORRECT (but lossy)

Full suffix adds 1.096 bits (12.0%) over sfx_fam. The VP (vowel prefix) carries 45% of suffix entropy and most section information. sfx_fam discards this.

**Verdict:** sfx_fam is the right abstraction for grammar (T1/T2 generalises better). For section-specific content analysis, use full suffix or VP+terminal decomposition directly from the pkl.

---

## Final Architecture

```
Layer 0:  enriched_records.pkl       GROUND TRUTH        37,465 tokens
Layer 1:  p70_rules_canonical.json   CHARACTER GRAMMAR    210 rules
Layer 2:  p70c_pos_layer.pkl         SLOT+POSITION        6,750 quints
Lookup:   transition_lookup.json     SEQUENTIAL FILTER    8×8 matrix
Metadata: section, quire, para_flag  FOLIO/LINE CONTEXT   per-record
```

### Information Budget

| Source | MI (bits) | % of H(quad) |
|--------|----------|-------------|
| Section | 0.810 | 8.88% |
| Position | 0.518 | 5.68% |
| prev_sfx (lookup) | 0.757 | 8.30% |
| Para flag | 0.030 | 0.33% |
| Quire | 0.518 | 5.68% |
| **Total explained** | **2.634** | **28.87%** |
| **Unexplained (lexicon)** | **6.490** | **71.13%** |

The 71.13% unexplained is the content — which specific tokens are used where. That's the lexicon, not the grammar. Any notation system must leave this gap open.
