# VBM Joachim-exact v9 — Q0b published-atom source correction

Date: 2026-09-01
Status: **FROZEN AFTER Q0, BEFORE Q0b STATISTICS AND BEFORE ANY Q1 LANGUAGE FIT**

## Why this exists

Q0 was intentionally frozen from the newly supplied f115v.26 example alone and therefore admitted only the explicitly visible multi-character left halves `qo` and `ch`.

After Q0 completed, a source check recovered a rule Joachim had published independently on 11 July 2026, before this experiment: in his core analysis, `cth`, `ckh`, `cph`, `cfh`, `ch`, `sh`, and `qo` are treated as separate atomic / one-glyph units. This is external prior art, not a pattern inferred from Q0 outcomes.

Q0 remains binding and is not rewritten. Q0b is a stricter source-fidelity requirement: Q1 may open only if the same structural gates also pass when the first bridge half is parsed using Joachim's fuller published atomic inventory.

Source: Voynich Ninja, thread 5312, post dated 11 July 2026 (JoJo_Jost): `cth, ckh, cph, cfh, ch, sh, qo` treated as separate atomic / one glyphs.

## Parser change and only parser change

Primary Q0 rules remain unchanged except `L(w)` now uses longest-prefix matching over this frozen ordered atomic set:

`ckh, cth, cph, cfh, ch, sh, qo`

If none matches, `L(w)` is the first character.

Right half remains the final character. The entire substring between left and right halves remains one nucleus. Empty nuclei remain allowed. `SINGLE_SHARED` remains primary. No additional atoms may be added.

The f115v.26 fixture must still parse identically to Q0.

## Data, split, exclusions, measurements and gates

Identical to `VBM_JOACHIM_EXACT_V9_Q0_PROTOCOL.md`:

- ZLZI;
- H1 and C1 unread and excluded;
- exact same deterministic TRAIN / INTERNAL_HOLDOUT folio split;
- at least 500 held-out nucleus and bridge events;
- TRAIN nucleus occurrence coverage >= 0.90;
- TRAIN bridge occurrence coverage >= 0.97;
- fixture exact = 100%.

No threshold or split changes are allowed.

## Binding rule

Q1 may be preregistered only if **both Q0 and Q0b pass**. If Q0b fails, the source-faithful v9 model stops before language fitting even though Q0-minimal passed.
