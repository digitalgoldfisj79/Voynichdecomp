# Data Files

## Ground Truth

### enriched_records.pkl
37,465 VMS tokens, each a dictionary with 4-slot PGCS decomposition plus metadata.

```python
{
  'token': 'qokeedy',        # EVA token string
  'prefix': 'qo',            # 8 values: ∅, ch, d, o, qo, s, sh, y
  'gallows': 'k',             # 9 values: ∅, cph, ckh, cth, f, k, p, t, cfh
  'core': 'eed',              # raw core string
  'suffix': 'y',              # raw suffix string
  'sfx_fam': 'Y',             # 7 families: BARE, L, M, N, OTHER, R, Y
  'm_core': 'ee',             # minimal core (suffix-stripped)
  'section': 'Herbal-A',      # 9 sections
  'folio': 'f1r',
  'line_no': 1, 'pos': 3,
  'is_first_word': False,
  'is_last_word': False,
  ...
}
```

**Source:** ZLZI transcription (ZL_ivtff_2b) from voynich.nu.
**Authority:** This file is the ground truth. All other layers derive from it.

### voynich_section_map.json
Folio → section mapping for all VMS folios. 9 canonical sections:
Herbal-A, Herbal-B, Astronomical, Cosmological, Zodiac, Rosettes, Balneological, Pharmaceutical, Stars.

Includes `old_to_new` mapping from traditional section names.

## Character Grammar

### p70_rules_canonical.json
210 character adjacency rules. Schema: `P70-canonical-sections`.

- 81 chargram rules (bigram patterns)
- 36 prefix rules
- 41 suffix rules
- 52 pair rules (cross-word-boundary)

Each rule: `{rule_id, kind, pattern, pred_side, base_weight, allow, deny, w_by_section}`.
Coverage: 92.96% character-level, 71.93% full-word, 99.87% any-rule.

## Constraint Layer

### p70c_full_layer.pkl
Pre-built `P70C_Full` instance. Requires `src/p70c_full.py` on import path to unpickle.

- 5,172 quads (P, G, mc, SF)
- 6,750 quints (P, G, mc, SF, Position)
- 8 transition distributions (prev_sfx → prefix)
- 16 quire profiles
- 6 production cluster assignments
- Full suffix access per quad/quint

### p70c_full_spec_v1.json
JSON export of the full layer. 6,750 entries, each:

```json
{
  "prefix": "ch", "gallows": "∅", "m_core": "∅", "sfx_fam": "Y",
  "position": "MID",
  "count": 1362, "tier": "T1", "prob": 0.036354,
  "full_suffixes": ["chy", "dy", "edy", "eedy", "eey", "ey", "ody", "shy", "y"],
  "examples": ["chchy", "chdy", "chedy", "cheedy", "cheey"]
}
```

Plus `transition_lookup` (8×8 matrix) embedded in the spec.

### transition_lookup.json
Standalone 8×8 matrix: P(prefix | prev_sfx_fam).
8 conditioning values: LINE_START, BARE, L, R, N, Y, OTHER, M.
8 prefix targets: ch, d, o, qo, s, sh, y, ∅.

Key transitions: Y→qo (26%, 1.9×), BARE→∅ (39.5%, 1.8×), LINE_START→y/d enriched.
