# M19 German-map post-selection transfer diagnostic

Date: 2026-08-09
Status at freeze: no TTLI/VDRB or lexical result for the v0.9 German map has been observed.

## Status

This is explicitly **exploratory/post-selection**. German was selected because v0.9 ranked it first on held-out ZLZI with 99.19% independent-fit agreement, while missing the frozen primary forward-margin threshold (0.02972 < 0.05). This diagnostic cannot retroactively turn v0.9 positive.

## Frozen object

Use the exact better v0.9 German ZLZI glyph→M19-number mapping unchanged:

`a→5, b→22, c→6, d→4, e→1, f→16, g→22, h→3, i→10, j→20, k→2, l→12, m→9, n→23, o→1, p→7, q→4, r→24, s→30, t→8, u→0, v→28, x→28, y→5, z→20`.

No fitting, remapping, key completion or language-specific modification is permitted.

## Data

Use exactly the v0.9 45 held-out folios determined by namespace `20260809|M19HMMv09|folio`.

Evaluate three renderings of those same folios:
- ZLZI (lexical diagnostic only; forward ranking already known),
- TTLI,
- VDRB.

Language models are exactly the v0.9 fresh models trained on sentence residues `{3,4,8,9}` mod 10.

## Tests

For TTLI and VDRB independently:
1. apply the literal frozen ZLZI glyph→number map to shared glyph labels;
2. unknown glyph labels make the containing word unavailable for forward/lexical scoring; report mapped-letter coverage;
3. rank all eight frozen languages by exact M19 hidden-letter forward likelihood;
4. report German forward margin over runner-up;
5. Viterbi-decode under German only and compute dictionary-hit fraction;
6. compare that fraction with 128 seeded permutations of the frozen number assignment and report lexical z.

Also compute ZLZI held-out German lexical z under the same frozen map, without inspecting individual decoded words.

## Exploratory survival criterion

The selected German lead `SURVIVES TRANSCRIPTION DIAGNOSTIC` only if BOTH TTLI and VDRB satisfy the v0.9 transfer thresholds unchanged:
- mapped-letter coverage >=0.90;
- German ranks first by exact forward likelihood;
- German forward margin >=0.03 nats/letter;
- German lexical z >=3.

Additionally ZLZI German lexical z must be >=5, matching the untriggered v0.9 candidate lexical gate.

Failure of any criterion = `POST-SELECTION LEAD DOES NOT SURVIVE`.

Passing is not confirmation. It would justify a new German-only confirmatory experiment on a fresh folio split/transcription surface.