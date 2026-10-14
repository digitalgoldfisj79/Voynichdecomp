# Voynich Super-Grammar v04 — portable realization-operator library

- key: `voynich_supergrammar_v04_realization_operator_library_20260924`
- status: active
- created: 2026-09-24T10:20:15.148208+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---


# Super-Grammar v04 realization-operator extension — 2026-09-24

Status: PROVISIONAL_CORE_EXTENSION; not yet a certified theorem.

## Main result
An exhaustive one-edit graph over all 5,076 exact running-text EVA types was built. Operator classes were discovered only in Herbal A versus Herbal B and then replayed on non-Herbal Pharmaceutical-A versus Biological+Stars-B. Twenty-one context-defined operators preserve A/B polarity and clear >=2 null SD in both domains.

Strong portable examples:
- insert d pre-final before y: discovery effect 2.717, null SD 0.395, ratio 6.87; validation effect 1.417, null SD 0.294, ratio 4.83.
- substitute a/o pre-final: -1.595 / 0.419 = 3.81; validation -2.001 / 0.377 = 5.31.
- substitute e/o internally: -2.491 / 0.684 = 3.64; validation -2.527 / 0.550 = 4.59.
- insert o internally before d: -3.131 / 0.791 = 3.96; validation -3.613 / 0.822 = 4.40.
- substitute d/e pre-final: -3.441 / 0.644 = 5.34; validation -1.881 / 0.476 = 3.95.

## Compression consequence
Eight highest-coverage externally portable operators:
INS:d:PRE_FINAL:R=y; SUB:e>o:INTERNAL; SUB:a>o:PRE_FINAL; SUB:d>e:PRE_FINAL; INS:o:INTERNAL:R=d; SUB:m>r:FINAL; SUB:l>o:INITIAL; SUB:d>k:INITIAL.

They reduce 5,076 exact types to 3,815 formal families (24.84%), touch 65.92% of token occurrences, and reduce raw hapax types from 3,375 to 2,594 (23.14%).

Adding the separately supported gallows k/t and p/f substitution families gives 3,394 formal families (33.14% exact-type reduction), touches 74.05% of token occurrences, and leaves 2,328 family hapaxes (31.02% hapax reduction).

All 21 externally validated regime operators provide a broader upper bound of 35.20% type reduction before gallows; this is not adopted as the conservative canonical compression because component chaining grows with library size.

## Architecture
regional/page state -> line/layout state -> token-family selection -> portable regime-conditioned realization operator -> local-state glyph-family choice where applicable -> internal legality -> exact surface form -> junction/memory layers.

## Firewall
These are formal surface families. No semantic equivalence, plaintext synonymy, decipherment, or German provenance follows from component membership. SG64 extends SG47; it does not erase SG12's macro identifiability bound. Full alternate-transcription and repository clean-room replay remain open.


## SG66 — latent-root obfuscation hypothesis
The stronger same-root hypothesis now has positive evidence.
- Raw Currier A/B exact-token JSD 0.46057 -> 0.32460 under conservative top8 collapse (-29.52%); excess reduction over component-size-matched random collapse = 14.17 null SD.
- Top8+gallows -> 0.30624 (-33.51%); excess = 9.24 null SD.
- Across five strongest portable operators, paired variants are 9.45 null SD more externally context-similar than matched one-edit controls.
- Strong same-root candidates: Xy/Xdy (5.59 SD context excess), pre-final d/e (25.44 SD), pre-final a/o (2.67 SD).
- Unresolved as same-root: internal e/o (1.77 SD) and internal o insertion before d (1.46 SD).

Working interpretation: some realization operators may disguise one latent root; others may encode genuine functional differences. This is not yet semantic decipherment.

