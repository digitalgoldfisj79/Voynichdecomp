# BnF 7342 numerical-alphabet key-granularity ladder v0.4 — preregistration

Date: 2026-08-09
Parent result: `experiments/bnf_onomancy_global_key_v0_3/RESULT.md`
Status at freeze: NO v0.4 Voynich language scores observed.

## Question

The v0.3 global fixed-key T2 model was recoverable on synthetic controls but rejected on Voynich. Does the same historically narrow BnF-derived homophonic substitution class become resolving if the fixed key changes only at a coarse manuscript-production boundary?

This is a bounded relaxation ladder, not an unconstrained search for a readable output.

## Fixed cipher model

Primary cipher model remains T2 from BnF lat. 7342: the four globally injective table-pairs `FM`, `FG`, `FL`, `ML`, giving four marked homophones per 23-letter plaintext alphabet symbol. Voynich literal transliteration glyphs are opaque cipher symbols. Spaces are preserved as word boundaries. Each key maps each observed Voynich glyph label to exactly one plaintext letter and respects <=4 cipher homophones per plaintext letter within that key.

T3 is not allowed to establish a primary signal in v0.4.

Plaintext alphabet and normalization remain v0.3: `abcdefghiklmnopqrstuxyz`, with `j→i`, `v/w→u`.

Language panel remains frozen: Latin, Italian, German, French, Greek, Hebrew, Arabic, Spanish/Castilian.

Primary transcription: ZLZI. Confirmation transcriptions, only if a candidate triggers: TTLI and VDRB.

## Frozen metadata sources

- `main/voynich_transcriptions_slim.json`
- `main/daiin_manifest.csv` for folio-level Currier class (`A/B/C`) and quire
- `main/voynich_section_map.json` for content section

No section/Currier/quire assignment may be changed after a language score is observed.

## Ladder and stop rule

Evaluate in this order:

1. **K-CURRIER** — one key per `A`, `B`, `C`.
2. **K-SECTION** — one key per frozen content section (`Herbal-A`, `Herbal-B`, `Astronomical`, `Cosmological`, `Zodiac`, `Rosettes`, `Balneological`, `Pharmaceutical`, `Stars`, `text-only`).
3. **K-SECTION×CURRIER** — one key per observed section×Currier cell.
4. **K-QUIRE** — one key per quire.

A rung is eligible for Voynich scoring only if its positive-control qualification gate passes. If a rung produces a fully confirmed primary signal, stop and report; do not continue to a more flexible rung. If a rung fails its control gate, mark it UNDERPOWERED and continue to the next rung only if that next rung is independently gateable.

No folio-level, paragraph-level, line-level, token-level or free context-conditioned keys are permitted in v0.4.

## Folio split

For every grouping, assign entire folios to train/hold deterministically by SHA-256(`20260809|folio|grouping`). Target 80/20 within each group, with at least one hold folio for groups with >=2 folios. A group with one folio has no admissible held-out key test and makes that rung fail qualification unless the group is excluded prospectively by the rule below.

### Tiny-group rule

Before any language optimization, a group is `EVALUABLE` only if it has:
- >=3 ZLZI folios,
- >=1500 total non-space alphabetic glyph positions,
- >=500 held-out glyph positions under the deterministic split.

Non-evaluable groups are excluded from both controls and Voynich scoring for that rung and their excluded glyph count is reported. A rung is admissible only if >=80% of all ZLZI glyph positions fall in evaluable groups.

## Positive-control qualification

For each rung, construct four synthetic piecewise-key controls: Latin, Italian, German, Hebrew. Use held-out text from the same frozen language corpora as v0.3. Mimic the exact ZLZI evaluable-group page lengths and train/hold assignments. Each group receives an independently randomized T2 key. Keys are globally fixed within group across its pages.

For each control, fit each group key on its synthetic training pages only and evaluate unchanged on synthetic held-out pages.

A rung PASS requires all:

- **P1 language identification:** correct language ranks first in 4/4 controls.
- **P2 weighted plaintext accuracy:** median held-out character accuracy >=0.90.
- **P3 worst-control accuracy:** minimum held-out character accuracy >=0.75.
- **P4 group coverage:** for each control, >=80% of evaluable-group held-out characters belong to groups whose own character accuracy >=0.70.
- **P5 mapping separation:** median whole-rung mapping-permutation z >=10.0.

A failed rung cannot yield historical evidence regardless of its Voynich score.

## Voynich scoring

For each admissible rung and each frozen language:

1. Fit one T2 key per evaluable group using ZLZI training folios only.
2. Apply those keys unchanged to held-out ZLZI folios in the same groups.
3. Compute held-out 4-gram mean log likelihood.
4. Compute a whole-rung mapping-permutation null by independently permuting each fitted group mapping within its observed capacity structure, preserving all ciphertext order and group boundaries. Use 64 null maps. Report z.
5. Compute a lexical sanity score: fraction of decoded held-out word tokens occurring in that language corpus vocabulary after the same normalization; compare against 64 mapping-permutation nulls and report lexical z.
6. Compute an MDL-style key penalty for comparison with the v0.3 global model: `25 * (K-1) * ln(23)` nats for K evaluable keys. Report held-out log-likelihood gain minus this penalty; this is diagnostic and cannot by itself establish a signal.

## Primary signal criterion

A ZLZI rung has a candidate only if one language satisfies all:

- mapping-permutation **z >= 10.0**;
- z-margin over second-ranked language **>=5.0**;
- lexical **z >=5.0**;
- at least 80% of evaluable held-out characters are in groups whose individual held-out mapping score is above the median of that group's permutation null;
- capacity respected in every fitted group.

If no language satisfies all, rung verdict = NO SIGNAL.

## Cross-transcriber confirmation

Only a ZLZI candidate is transferred. Apply the literal glyph→plaintext mappings, group by group, unchanged to TTLI and VDRB on folios shared with ZLZI. Unknown glyphs are omitted and omission fraction reported.

Confirmation requires, independently in both TTLI and VDRB:
- candidate language remains rank 1 among the eight-language panel;
- mapping-permutation z >=7.0;
- lexical z >=3.0;
- >=90% of evaluated glyph positions use mapped shared symbols.

A ZLZI-only candidate that fails either confirmation is `TRANSCRIPTION-DEPENDENT`, not a positive result.

## Interpretation classes

- `CONFIRMED PIECEWISE-KEY SIGNAL`: positive-control PASS + ZLZI primary criterion + TTLI PASS + VDRB PASS.
- `ZLZI CANDIDATE / TRANSCRIPTION-DEPENDENT`: ZLZI criterion passes but confirmation fails.
- `NO SIGNAL`: controls pass but no ZLZI language meets primary criterion.
- `UNDERPOWERED`: positive-control gate fails.

## Scope

Even a confirmed result would show compatibility with this cipher class, not identify BnF lat. 7342 as the source or prove historical transmission. A negative result rejects only the tested coarse-key T2 models with preserved spaces and the frozen language/normalization panel.
