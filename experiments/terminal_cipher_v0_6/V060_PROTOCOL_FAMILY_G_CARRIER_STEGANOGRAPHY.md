# v0.6 Family G — bounded carrier steganography

Date: 2026-07-16

Status: **FROZEN BEFORE IMPLEMENTATION OR RESULTS**

No test data or Voynich text has been inspected.

## Scope and falsifiability

Steganography is unbounded if an extraction rule may be invented after inspecting the manuscript. Family G therefore covers only four finite carrier classes that can be specified independently of Voynich outcomes:

1. **Acrostic/telestic:** first or last symbol of each line, paragraph or token sequence.
2. **Fixed token-position:** first or last symbol of token `k` within each line, for `k = 1..5`.
3. **Regular null extraction:** every `n`th token or every `n`th character, for `n = 2..12`, with offsets `0..n-1`.
4. **Repeated grille:** one fixed mask repeated over line-normalised rectangles of width 4–12, with mask density 10–40% and at most eight selected cells per period.

No diagonal, spiral, astronomical, illustration-conditioned, folio-specific or semantically selected path may be added. No rule may use Voynich content to determine its own parameters.

## Scientific target

Family G tests whether a meaningful linguistic payload embedded in a structured cover can be:

- detected against matched no-payload covers;
- assigned to the correct carrier class and parameters;
- recovered at useful accuracy;
- distinguished after correcting for the complete preregistered candidate search.

This is a carrier-detection programme, not an unrestricted search for readable fragments.

## Cover generators

Use the four frozen v0.5 control families:

- Markov-2;
- motif;
- copy-mutate;
- slot generator.

Covers must match payload and null examples in length, line-length distribution, alphabet, token-length distribution and generator family.

For embedding, the cover generator is constrained only at selected carrier positions. All unselected positions are sampled normally. Failed constraints are resampled without modifying the carrier rule.

## Payloads

- source languages: the six frozen UD languages;
- development plaintext length: 96 payload characters;
- locked-test payload length: 128 characters;
- payload source chunks are disjoint across train, development and test;
- payload may be monoalphabetically relabelled by a fresh key, but encrypted and unencrypted payload arms are reported separately;
- no semantic phrase or crib is supplied.

## Candidate enumeration

Every preregistered extraction candidate is evaluated on every cover. The exact candidate inventory is frozen before data generation.

Each extraction receives:

- train-only character-LM likelihood;
- compression/MDL score;
- recurrence and alphabet-utilisation penalties;
- best fresh monoalphabetic decipherment score where applicable.

The family-level statistic is the maximum candidate score after subtracting a null-calibrated search penalty estimated from matched no-payload covers. The raw highest linguistic-looking extraction is never treated as evidence without this correction.

## G1 — oracle carrier recovery

Supply the true carrier class and parameters, but hide whether the payload is plaintext or freshly monoalphabetically substituted and hide the substitution key.

Gate across 64 English development covers (4 cover generators × 4 carrier classes × 4 replicates):

- mean payload recovery at least 90%;
- minimum at least 70%;
- at least 58/64 covers recover at least 85%;
- encrypted-payload recovery mean at least 85%.

Failure closes Family G because the payload cannot be recovered even when the carrier is known.

## G2 — blind carrier detection and recovery

The solver receives only the cover, line boundaries and tokenisation. It must search the complete frozen candidate inventory and either:

- return one carrier rule and recovered payload; or
- abstain.

Development set:

- 64 payload covers;
- 256 matched null covers;
- cover-generator identities hidden from the detector.

Gate:

- family-level payload-detection AUROC at least 0.95;
- false-positive rate at the frozen operating point at most 1%;
- carrier-class accuracy at least 85% among detected payload covers;
- exact parameter accuracy at least 75%;
- mean recovered payload accuracy at least 80%;
- at least 54/64 payload covers recovered at 70% or better;
- abstention permitted but counts as a miss on payload covers.

One development-only amendment is permitted. It may alter score calibration, classifier architecture or monoalphabetic search budget, but not the carrier inventory, generators, payloads, gates or split.

## Locked test

A passing G2 system is frozen and evaluated once on:

- 80 untouched payload covers;
- 320 untouched matched null covers;
- six languages;
- payload length 128;
- fresh carrier parameters and fresh substitution keys.

No post-test modification is permitted.

## Voynich admissibility rule

Voynich application is forbidden unless the locked test passes. Any candidate carrier must then:

- recur under the same rule across held-out folios;
- recover payload with stable language-model evidence;
- survive family-wise correction across all candidates and representations;
- outperform matched structured-generation controls;
- produce no comparable payloads under shuffled line, token and folio nulls.

A single readable fragment, one folio-specific rule or an extraction selected after inspection is inadmissible.

## Closure rule

If G2 fails after its one amendment, or its locked test fails, bounded carrier steganography is closed. Arbitrary steganography remains logically possible but is classified as non-identifiable and cannot remain an empirical working explanation.