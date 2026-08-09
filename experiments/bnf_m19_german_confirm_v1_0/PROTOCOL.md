# BnF M19 fixed-German-map fresh-folio confirmation v1.0 — preregistration

Date: 2026-08-09
Status at freeze: none of the v1.0 fresh folios has been scored under the selected German map.

## Selection history

v0.9 was a qualified eight-language test. German ranked first on held-out ZLZI with 99.19% independent key agreement but missed the preregistered language-margin gate (0.02972 < 0.05), so v0.9 remained negative. A labelled post-selection diagnostic then transferred the exact frozen German map unchanged to TTLI and VDRB on those same held-out folios and passed all exploratory transfer/lexical thresholds.

v1.0 is the first confirmatory test of that selected German hypothesis on folios that were **never used by v0.9 for either mapping fitting or held-out ranking**.

## Frozen German M19 key

No refitting or modification is permitted:

`a→5, b→22, c→6, d→4, e→1, f→16, g→22, h→3, i→10, j→20, k→2, l→12, m→9, n→23, o→1, p→7, q→4, r→24, s→30, t→8, u→0, v→28, x→28, y→5, z→20`.

This is a valid exact M19 map: all 19 BnF unmarked values occur; exactly six values have a second surface homophone; no value has multiplicity >2.

## Fresh Voynich panel

Reconstruct the deterministic v0.9 split exactly. Let:
- `T09` = the 59 ZLZI folios used to fit the v0.9 mappings;
- `H09` = the 45 ZLZI folios used for v0.9 held-out ranking and the post-selection transfer diagnostic.

The v1.0 confirmation panel is:

`C10 = all ZLZI-transcribed folios - T09 - H09`.

Expected size is 122 folios; the runner verifies rather than assumes this. Any overlap with T09 or H09 is a hard failure.

No map, language or threshold may depend on C10 content.

## Language models

Use exactly the v0.9 language models trained on UD sentence residues `{3,4,8,9}` mod 10 for the frozen eight-language panel:
Latin, Italian, German, French, Ancient Greek, Hebrew, Arabic, Spanish/Castilian.

No language model is retrained after C10 is scored.

## Three transcription surfaces

Evaluate the fixed key on C10 independently under:
- ZLZI;
- TTLI;
- VDRB.

For TTLI/VDRB, use every C10 folio on which that transcription has text. Unknown literal glyph labels make their containing word unavailable to forward/Viterbi scoring; mapped-letter coverage is reported exactly.

## Primary language confirmation

For each transcription surface:
1. apply the frozen glyph→number map unchanged;
2. compute exact hidden-letter M19 forward nats/letter for all eight languages;
3. rank languages;
4. report German margin over runner-up.

PASS thresholds:
- ZLZI: German rank 1 and margin >= **0.05 nats/letter**;
- TTLI: German rank 1 and margin >= **0.03**;
- VDRB: German rank 1 and margin >= **0.03**;
- mapped-letter coverage: ZLZI >=0.99; TTLI/VDRB >=0.90.

These are the unchanged v0.9 primary/transfer margins.

## Lexical confirmation

For each surface, Viterbi-decode mapped words under the frozen German language model and exact BnF emission law. Compare German dictionary-hit fraction against 256 seeded permutations of the fixed M19 number assignment.

PASS thresholds:
- ZLZI lexical z >= **5**;
- TTLI lexical z >= **3**;
- VDRB lexical z >= **3**.

No decoded word strings may be inspected before all primary and lexical gates are evaluated.

## Internal distribution check

Before any decoded words are exposed, partition C10 ZLZI folios into four deterministic hash buckets using `20260809|M19GermanConfirm|bucket|folio`. Apply the same fixed key and rank all eight languages separately in each bucket.

Robustness requirement:
- German rank 1 in at least **3 of 4** buckets;
- median German-v-runner-up margin across buckets >0.

This guards against the whole-panel result being driven by one content cluster. It is binding for confirmation.

## Verdict

`CONFIRMED FRESH-PANEL GERMAN M19 SIGNAL` requires every primary, lexical and bucket criterion above.

Otherwise verdict is `GERMAN M19 LEAD FAILS FRESH CONFIRMATION`.

A PASS establishes a reproducible statistical compatibility of fresh Voynich text with the **selected fixed M19/German model**. It still does not prove that Voynich is German, that BnF lat.7342 was used, or that the historical encoder used this exact mechanism.

## Post-gate readable output

Only after a full PASS, the runner may output the first 100 Viterbi-decoded ZLZI word tokens from C10 in canonical manuscript folio/line/token order, without filtering or hand-selection, together with dictionary-hit flags. This sample is diagnostic, not an additional gate.