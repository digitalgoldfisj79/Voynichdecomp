# Recoverability frontier v0.5.1 — protocol amendment A

Date: 2026-07-15

Status: fixed before any v0.5.1 execution.

## Numeric-label invariance correction

The v0.5.0 `polyalphabetic` and `feedback` generators perform modular arithmetic directly on integer symbol identifiers. Those integer identifiers have no observable ordering when ciphertext units are arbitrary glyphs. A solver that receives the raw integers could exploit an artificial property that would disappear under an independent surface renderer.

First-occurrence canonicalisation correctly removes this shortcut, but it also means that simple period/shift or key/state search is no longer a valid solver. Such a solver would additionally need to infer the hidden surface permutation.

Therefore the initial v0.5.1 execution is restricted to `mono`, whose substitution structure is invariant under arbitrary relabelling. No result from raw numeric IDs will be reported.

## Revised staged gate

Stage 1A evaluates monoalphabetic substitution only across all six languages, three lengths and unseen keys.

Proceed to homophonic and null-homophonic solvers only if:

- mean monoalphabetic character accuracy is at least 70%;
- every language exceeds 50%;
- exact recovery and runtime are reported;
- the result uses first-occurrence canonicalised cipher symbols.

Polyalphabetic and feedback families remain in the benchmark, but require a later hidden-renderer-aware solver or a revised operator definition that makes the assumed alphabet ordering explicit and scientifically defensible.
