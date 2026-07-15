# Recoverability frontier v0.5.2 — protocol amendment D

Date: 2026-07-15

Status: fixed before pair-block development execution.

## Motivation

The true key strongly outscored all recovered English 384-character keys, including searches supplied with the exact observed homophone inventory. Length-proportional annealing temperature made performance worse. The remaining defect is therefore the proposal topology.

A homophone group may require several cipher symbols to change plaintext assignments together. Single-symbol moves traverse low-scoring intermediate states and become trapped.

## Frozen pair-block move

For every selected pair of plaintext labels:

1. collect all cipher symbols currently assigned to either label;
2. enumerate every joint binary reassignment of that small block;
3. retain assignments satisfying both labels' multiplicity caps;
4. choose the highest-scoring block assignment under the unchanged train-only quadgram objective.

Because the benchmark multiplicity cap is at most four, a pair contains at most eight cipher symbols and requires at most 256 exact evaluations.

The development search uses repeated randomized sweeps over label pairs and multiple bounded random restarts. It does not receive the true inventory, key or plaintext.

## Development gate

Run English and Hebrew development data at 384 characters only. Proceed to a new untouched test block only if:

- English mean recovery is at least 70%;
- Hebrew mean recovery is at least 70%;
- both improve over the failed single-symbol development regime;
- schedule selection uses development data only.

The quadgram model, multiplicity caps and recovery metric remain unchanged.
