# Demolition brief — find what is wrong with these findings

Paste this into a new chat with no other context. Attach or allow access only to the files listed under **Data**.

## Instruction

Find what is wrong with each finding below. Assume nothing in it is true. For each one, attempt a concrete demolition: a bug, a confound, a null that is too easy, a selection effect, a representation artefact, or a number that does not reproduce. Re-derive the numbers yourself where you can. Do not propose improvements until you have tried to break it.

For each finding, return one line:
`<ID> | SURVIVES / DAMAGED / DEMOLISHED | the strongest attack you made | the number you obtained, if any`

## Data

Only these files are allowed. Do **not** open any `REVIEW_*`, `RELEASE_*` or handoff document; they contain the authors' reasoning.

Repository `digitalgoldfisj79/Voynichdecomp`, branch `supergrammar-v031-candidate`:
- `voynich_transcriptions_slim.json` — sha256 `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`
- `research/supergrammar_v031_certifier_20260928.py` — produces F1, F2, F3, F4, F5, F6
- `research/supergrammar_v031_result_20260928.json` — its full output
- `research/supergrammar_v03_conditionals_20260920.py` — produces F7
- `research/supergrammar_v03_closure_certifier_20260920.py` — produces F8
- `research/supergrammar_v03_r64_current_20260920.py` — produces F9

Reproduce F1–F6:

`python research/supergrammar_v031_certifier_20260928.py --corpus voynich_transcriptions_slim.json` (about 70 s on one core).

**Corpus and layers.** Paragraph-text lines of Voynich running text in six EVA-family transliterations: ZLZI (34,087 tokens, 204 folios), ZLZB, TTLI, JSLI (7,315 tokens, 91 folios), VDRB, TTIA.

**Folds.** Five folds of whole bifolia.

**Null.** Unless stated otherwise, the null SD is a sign-flip SD over per-bifolium mean gains. The ratio is |effect| / null SD, and ≥2 counts as resolved.

## Findings

**F1 — Order-2 context.** A second preceding within-token glyph adds held-out predictive information beyond the immediately preceding glyph.
- Tested in all 6 layers, both reading directions, and smoothing α ∈ {4, 16, 64, 256}.
- Worst cell: effect 0.231214 bits/glyph, null SD 0.040103, ratio 5.77. All 48 cells resolve positive.

**F2 — Order-3 is not robust.** Third-order within-token context does not give a smoothing-robust gain.
- 23 of 48 cells fail.
- Worst cell: VDRB α=4 left-to-right, effect −0.04482, null SD 0.00786.

**F3 — 19 structural zeros.** These token-internal bigrams never occur in any of the 6 layers: ci dh dn kk km kn kp lh ln pl pn pp pt tm tn tp tr tt yn.
- Expected counts come from a null that preserves folio × within-token position × token-length bin.
- Raw EVA: weakest ratio 3.55 (pn: expected 10.44, null SD 2.94).
- Atomic frame (ch sh cth ckh cph cfh merged into single units): weakest ratio 2.70 (lh: expected 6.65, null SD 2.46).
- The 19 were selected earlier from a screen of 77 zero-count bigrams in the same corpus.

**F4 — SPACE edge dependency.** Across a within-line space, the final glyph of the previous token adds held-out information about the initial glyph of the next token, beyond section, hand, position class and length bin.
- ZLZI worst: 0.03810 / 0.01543 = 2.47.
- JSLI α=4: −0.03594 / 0.01665.

**F5 — SPACE > LINE_BREAK.** The same edge gain is larger across within-line spaces than across physical line breaks, in ZLZI, ZLZB, TTLI, VDRB and TTIA.
- Full volume: worst cell over all 6 layers 0.12958 / 0.04474 = 2.90. The five named layers score 5.40–5.71.
- Cross-paragraph junctions removed: 2.89.
- SPACE training volume subsampled to LINE_BREAK volume, 10 seeds: the five named layers pass 40/40 cells each. JSLI passes 14/40.
- LINE_BREAK gain is negative at α≤16.

**F6 — No independent line-boundary ED1 avoidance.** The rate of edit-distance-1 pairs across line breaks is not below a marginal-preserving page × vertical-quartile assignment null: −0.001822 / 0.001704.

**F7 — Conditional graphotactics.**
- (a) q-presence changes k/t choice beyond the exact continuation tail and section: ZLZI 0.023703 / 0.004965 = 4.77.
- (b) i-run length adds held-out information about the terminator beyond one and two preceding glyphs: weakest ratio 3.52 across layers.

**F8 — Previous-token residual.** The previous token's (first glyph, last two glyphs, length) adds a held-out transition gain beyond the parent model: ZLZI 0.007355 / 0.001914 = 3.84.

**F9 — Reuse channel.** In a generator, a banded same-page exact-reuse channel over lags 2–64 improves the lag-2–5 exact-recurrence error compared with no reuse: 0.005672 / 0.002538 = 2.23, 5 of 5 folds.

## Return

- The per-finding table.
- Any bug you found, with file and line.
- The one finding you would bet against.
