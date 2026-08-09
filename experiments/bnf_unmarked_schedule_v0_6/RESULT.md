# BnF 7342 unmarked numerical-value schedule programme v0.6 — result

Date: 2026-08-09
Branch: `experiment/bnf-unmarked-schedule-v0.6-20260809`
Protocol freeze: `25e45123d16e7f7c9af54200804e869c62367fd0`
Amendment 001: `f81c7dcaa465474dbe45870da4377a919bc5276a`
Runner: `e09575bc6babec172c88f20c2903a56619da47d6`
HF job: `6a781851da2af92a634efe33`

## Verdict

**ALL 25 FROZEN UNMARKED SCHEDULE+ROTATION MODELS STRUCTURALLY REJECTED.**

No language model or decoder was fitted. This is a purely combinatorial rejection of the tested assumption that each Voynich surface glyph carries one global unmarked numerical value while the active BnF table is supplied by one of five deterministic positional schedules.

## Calibration

Long disjoint controls were constructed from Latin, Italian, German and Hebrew. For every one of the five schedule families, the values present in all 20 language×rotation controls were the same 17-value expected repertoire:

`0,1,2,3,4,5,6,7,8,9,10,12,16,22,23,24,30`

Values 20 and 28 were not universal only because the frozen Hebrew romanized control did not emit them under these schedules; they were therefore not required for the Voynich structural gate.

Primary ZLZI corpus:
- 5,162 transcribed lines;
- 186,727 alphabetic glyph positions;
- 25 lowercased surface labels.

## Results

Every rotation of every schedule lacked at least one required numerical value in the union of legal global glyph assignments.

| schedule family | rotations passing / 5 | characteristic missing values |
|---|---:|---|
| CHAR_CONTINUOUS | 0 | usually 0,9,10,12,16,22,23,24,30 |
| CHAR_WORD_RESET | 0 | closest: rotation 0 lacks 0,23,30; rotation 1 lacks 22,24 |
| WORD_CONTINUOUS | 0 | usually 0,9,10,12,16,22,23,24,30 |
| WORD_LINE_RESET | 0 | usually 0,7/8,9,10,12,16,22,23,24,30 |
| LINE_CONTINUOUS | 0 | usually 0,9,10,12,16,22,23,24,30 |

The nearest cases were `CHAR_WORD_RESET` rotations 0 and 1, but even these could not represent all values that appear in every long positive-control plaintext.

## Why it fails

For most common Voynich glyph labels, the observed occurrences span all five scheduled table states. A global numerical value assigned to such a glyph must therefore belong to the intersection of the value sets of F, M, G, L and H, leaving essentially only the low common values. High/special values such as 22, 23, 24 and 30 occur in long ordinary plaintext controls but require table-specific support. No surface glyph has the necessary occurrence-phase restriction to carry the complete expected repertoire under the frozen schedules.

## Scope

This rejects only:

`global Voynich glyph → one unmarked number` + `deterministic positional table schedule`.

It does not reject hidden/free table choice, table-specific surface mappings, changing numerical alphabets, nulls, transposition, syllabic units, or non-linguistic generation.
