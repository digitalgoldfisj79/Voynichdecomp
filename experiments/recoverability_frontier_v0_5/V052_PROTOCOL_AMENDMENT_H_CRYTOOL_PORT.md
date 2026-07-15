# Recoverability frontier v0.5.2 — protocol amendment H: CrypTool-style search

Date: 2026-07-15

Status: fixed before execution.

## Closure of project-local search variants

The following project-local English 384-character development searches all failed the 70% gate:

- flexible single-symbol annealing;
- fixed-inventory exact pair blocks;
- flexible-inventory pair blocks;
- block/anneal hybrid;
- nested inventory beam;
- length-scaled temperature.

The nested inventory beam reached only 40.9180% and moved the inferred inventory farther from truth. Scientific SHA-256: `f8505568cf9b248d314a2f98d117b72861dfc3f20bebbdb4694b16f44140ed19`.

Further variants of the same local proposal family are stopped.

## Independent algorithm source

The next diagnostic ports the search architecture of CrypTool 2's `HomophonicSubstitutionAnalyzer`, pinned at commit:

`d7d754af55c167941bec7fb56e965f309d050a12`

Relevant Apache-2.0 sources:

- `CrypPlugins/HomophonicSubstitutionAnalyzer/HillClimber.cs`;
- `CrypPlugins/HomophonicSubstitutionAnalyzer/SimulatedAnnealing.cs`;
- `CrypPlugins/HomophonicSubstitutionAnalyzer/HomophonicSubstitutionAnalyzerSettings.cs`.

Original copyright notices name Nils Kopal and, for the simulated-annealing algorithm, George Lasry. The Python port retains attribution and is itself limited to the research benchmark.

## Material algorithmic differences

The port uses:

1. exhaustive deterministic sweeps over every pair of cipher-symbol assignments rather than random swap proposals;
2. linear cooling over a fixed proposal budget;
3. an acceptance-probability floor matching the CrypTool implementation (`0.0085`);
4. multiple independent restarts;
5. targeted reassignment of three rare cipher symbols after repeated sweeps without a new global best;
6. inventory limits derived from the bounded benchmark multiplicity caps;
7. first-occurrence-canonicalised ciphertext, preserving arbitrary label invariance.

The train-only quadgram objective, plaintext metric and corpus partitions remain unchanged.

## Temperature calibration

CrypTool's default absolute temperature is tied to its own language-statistics cost scale. The port therefore calibrates the initial temperature independently for each ciphertext from the median negative score change of deterministic sample swaps, targeting a frozen initial acceptance probability.

Development may select among three frozen step/restart schedules and two initial acceptance targets. No test data is used.

## Development scope and gate

Primary development environment:

- English;
- 384 normalized characters;
- 8 development source chunks and unseen keys.

Proceed to a fresh untouched test block only if mean recovery reaches at least 70%. Hebrew will be checked secondarily only after the English gate passes.
