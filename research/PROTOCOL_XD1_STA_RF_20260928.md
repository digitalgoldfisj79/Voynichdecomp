# XD1 STA/RF representation-sensitivity replay — frozen protocol

**Protocol ID:** XD1-STA-RF-20260928
**Frozen:** 2026-09-28 before computing the STA/RF repetition outcome.
**Parent:** XD1-CLOSEOUT-20260928.

## Purpose and limitation

Test whether the surviving Voynich within-line exact-repetition geometry depends on the EVA-family representation used in ZLZI.

Source: René Zandbergen RF 1b in STA1 level 0:
https://voynich.nu/data/sta/RF1b.txt

This is a **representation-sensitivity** test. RF is an automatically generated reference combining ZL and GC readings, so it is not an independent palaeographic transcription and must not be described as one.

## Corpus scope

Match the existing XD1 VMS arm:
- canonical Voynich folios from the frozen XD1 adapter;
- physical loci whose IVTFF locus type contains P;
- physical line boundaries are the IVTFF loci.

STA characters are two-byte codes matching [A-Z][0-9a-z].

Primary word convention:
- split only on definite IVTFF dot word separators;
- remove <-> uncertain-space markers, treating uncertain spaces as no boundary (long-word convention);
- extract the STA code sequence from each resulting word and compare exact code sequences.

Sensitivity:
- repeat after excluding any word chunk containing square-bracket uncertain-reading syntax.

No mapping from STA back to EVA is permitted in this test.

## Statistic

Identical to XD1-CLOSEOUT:
- lag 1 and lag 2 exact-word equality within physical line;
- null = independently permute the exact word multiset within the same line;
- 200 primary and 2,000 sensitivity permutations;
- analytic permutation expectation cross-check;
- fixed length bins ALL, 2-5, 6-10, 11-20, 21+;
- 2,000 stratified physical-line bootstrap replicates.

## Frozen representation gate

Criterion 5 ("independent representation sensitivity does not reverse") is operationalised here as:

PASS if, in both the primary and uncertainty-excluded STA arms:
1. lag 2 effect is positive and effect/null-SD >= 2; and
2. lag 1 is not significantly suppressive (signed effect/null-SD > -2).

Otherwise criterion 5 FAILS.

Passing this gate means only that the repetition result survives this materially different character representation/token-boundary rendering. It does not make RF independent of the ZL/GC source readings.

## Reporting

Every headline gives effect and null SD together.
If absolute effect/null-SD <2, lead with **the metric does not resolve this**.
