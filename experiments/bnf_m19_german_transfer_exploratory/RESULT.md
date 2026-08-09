# M19 German-map post-selection transfer diagnostic — result

Date: 2026-08-09
Protocol freeze: `bbfc3d6c25c19a36ba140f74ecfb9211620839a2`
Final runner: `413501cfd9ecc1b556debac26af99b6dfc3aefe1`
HF job: `6a78209ada2af92a634efe90`

## Verdict

**SURVIVES TRANSCRIPTION DIAGNOSTIC.**

This result is explicitly post-selection and does not retroactively make v0.9 positive. It establishes that the exact German M19 map selected after v0.9's near-miss transfers unchanged to two independent transcription traditions on the same held-out folios.

Frozen map:
`a→5, b→22, c→6, d→4, e→1, f→16, g→22, h→3, i→10, j→20, k→2, l→12, m→9, n→23, o→1, p→7, q→4, r→24, s→30, t→8, u→0, v→28, x→28, y→5, z→20`.

### ZLZI lexical gate
- coverage 100%;
- German Viterbi dictionary-hit fraction 0.19808;
- mapping-permutation null mean 0.08171, sd 0.02136;
- lexical z = **5.4474** (threshold 5).

### TTLI transfer
- mapped-letter coverage 100%, 28,618/28,618;
- German rank 1;
- forward score -2.684396 nats/letter;
- runner-up French -2.840967;
- margin **0.156571** (threshold 0.03);
- lexical fraction 0.19143 versus permutation-null 0.07535;
- lexical z **4.3071** (threshold 3).

### VDRB transfer
- mapped-letter coverage 100%, 29,388/29,388;
- German rank 1;
- forward score -2.703048;
- runner-up French -2.887523;
- margin **0.184475**;
- lexical fraction 0.19666 versus null 0.08069;
- lexical z **4.7429**.

All frozen exploratory survival conditions pass.

## Interpretation

This is the strongest lead produced by the BnF programme so far, but it remains selected after seeing v0.9 ZLZI results. The appropriate next step is not to inspect attractive decoded words. It is a genuinely fresh confirmatory test using folios never used by v0.9 for either mapping fit or held-out ranking, with the German numerical key fixed before those folios are scored.