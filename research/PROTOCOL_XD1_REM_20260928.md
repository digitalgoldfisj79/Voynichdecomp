# XD1 ReM historical-German extension — frozen protocol

**Protocol ID:** XD1-REM-20260928  
**Frozen:** 2026-09-28, before any ReM lag-1/lag-2 outcome from this programme.  
**Parent:** XD1-CLOSEOUT-20260928.

## Purpose

ReM is an independent diplomatic Middle High German control already used elsewhere in the Voynich programme. It is not selected from the current repetition outcomes.

This extension asks only whether the corrected Voynich within-line exact-repetition result is reproduced in a large real historical-language control with genuine physical lineation.

It is not a recipe/charm control and cannot replace the CoReMA recipe arm.

## Source

ReM v2.1 TEI from Zenodo record 13982324.
The canonical archive facts already verified in the project are:
406 documents, 2,236,137 recovered diplomatic tokens, 9,967,570 characters, 118 characters in the cleaned alphabet.

Extraction requirements:
- use text content of TEI <w>, never norm=;
- regroup *_mN subdivisions by base t-number to recover original word boundaries;
- preserve TEI physical <lb> lineation;
- use the existing frozen ReM cleaning regex.

Preflight FAILS unless the full-corpus recovered flat totals exactly match
406 / 2,236,137 / 9,967,570 / 118.

## Outcome-blind panel

Sort document IDs.
Require >=3,000 recovered diplomatic tokens.
Take the first 12 eligible documents.

This is the same selection rule previously used for the independent inverse-production confirmation panel. No genre or repetition outcome enters selection.

The preflight writes the selected physical lines and source/archive hashes to an atomic pickle and exits before scoring.

## Statistics

Identical primary statistic to XD1-CLOSEOUT:
- lag d in {1,2};
- observed exact equality token[i] == token[i-d] within physical line;
- null = independent permutation of the exact token multiset within the same physical line;
- 200 deterministic primary permutations;
- 2,000 deterministic sensitivity permutations;
- analytic exact permutation expectation cross-check;
- fixed length bins ALL, 2-5, 6-10, 11-20, 21+;
- 2,000 deterministic stratified physical-line bootstrap replicates.

Every headline reports effect and null SD together.
If |effect|/null SD <2, lead with: **the metric does not resolve this**.

## Additional clustering sensitivity

Compute lag-1/lag-2 statistics separately for each of the 12 documents.
Report the sign and observed/null ratio per document.
This is descriptive clustering sensitivity; no threshold is chosen after outcome.

## Frozen discriminator falsifier

The same pre-existing falsifier is applied descriptively:
a real-text control reproduces the proposed two-lag pattern if pooled lag-1 observed/null >=0.90 AND pooled lag-2 >=1.10.

A firing result retracts any claim that the two-lag pattern is absent from tested diplomatic historical German.
A non-firing result does not establish uniqueness beyond the tested controls.

## Stop / scope

Do not tune cleaning, line parsing, document selection, lag definitions, length bins, or null after scoring.
Any source-parser correction after seeing outcomes requires a new protocol version.
