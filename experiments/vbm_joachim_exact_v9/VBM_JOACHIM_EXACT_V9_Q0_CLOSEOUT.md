# VBM Joachim-exact v9 — Q0 / Q0b closeout

Date: 2026-09-01
Status: **Q0 PASS; Q0b PASS; H1/C1 UNREAD**

## Parser fixture

The supplied f115v.26 worked example parses exactly under both the minimal parser and the source-corrected published-atom parser. The supplied mappings reproduce the supplied continuous plaintext string exactly.

## Q0 minimal parser

TRAIN (excluding inherited H1 and C1 targets):
- 169 folios, 4,181 lines
- 28,050 valid tokens
- 24,634 non-empty nucleus events
- 23,741 bridge events
- 2,715 unique nucleus types
- 272 unique bridge types

INTERNAL_HOLDOUT:
- 43 folios, 1,094 lines
- 7,054 valid tokens
- 6,063 non-empty nucleus events
- 5,960 bridge events

Transfer:
- nucleus occurrence coverage: **0.922151** (5,591 / 6,063)
- nucleus type coverage: 0.6094
- bridge occurrence coverage: **0.991779** (5,911 / 5,960)
- bridge type coverage: 0.8446

All preregistered Q0 gates passed.

## Q0b published-atom correction

Externally published Joachim atom inventory applied by longest-prefix rule:

`ckh, cth, cph, cfh, ch, sh, qo`

TRAIN:
- 2,715 unique nucleus types
- 233 unique bridge types
- 1,783 nucleus singleton types
- 82 bridge singleton types

INTERNAL_HOLDOUT:
- 729 unique nucleus types
- 135 unique bridge types

Transfer:
- nucleus occurrence coverage: **0.9294079** (5,635 / 6,063; 428 unseen occurrences)
- nucleus type coverage: **0.60768**
- bridge occurrence coverage: **0.9911074** (5,907 / 5,960; 53 unseen occurrences)
- bridge type coverage: **0.85185**

All unchanged Q0 gates passed. Q1 synthetic calibration may therefore be preregistered.

## Capacity

For the source-corrected parser, the TRAIN surface dictionary contains 2,715 nucleus types and 233 bridge types.

Under the deliberately favourable v9-minimal codebook accounting:
- raw mapping-space capacity: about **67.8 kbits**
- optimistic minimum key cost (every nucleus mapped to a one-consonant value): about **20.9 kbits**

The large type tail is therefore a serious identifiability burden even though occurrence-weighted transfer is high.

### Supplied one-line feasibility fit

The f115v.26 example uses:
- 9 unique bridge mappings
- 8 unique non-empty nucleus mappings
- 19 consonant characters across those eight mapped nucleus values

Frozen favourable prefix code:
- bridge-map cost ≈ 20.90 bits
- nucleus-map cost ≈ 102.04 bits
- total ≈ **122.94 key bits** before type identifiers, parser/header cost, exceptions, or word-boundary information
- 30 produced plaintext characters
- ≈ **4.10 key bits per produced plaintext character**

This is not a significance test, but it makes the evidential burden explicit: a single fitted line can spend enough codebook freedom to be essentially non-diagnostic. The model becomes interesting only if the same assignments predict unseen material.

## Binding interpretation

Q0/Q0b establish only that the source-faithful whole-nucleus VBM has enough **occurrence-level dictionary reuse** to make held-out key transfer testable. They do not establish German, Bavarian, language, cipher, or plaintext.

The next gate is synthetic known-answer calibration. H1 and C1 remain sealed.
