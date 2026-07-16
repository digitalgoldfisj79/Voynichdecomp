# Recoverability frontier v0.5.5 — protocol amendment C: joint coordinate search

Date: 2026-07-16

Status: fixed after the frequency-screening diagnostic, before Stage B execution.

## Screening result

A frequency-derived substitution key is not a safe hard filter for transposition candidates.

Across 24 development trials:

- median true transposition rank: 11;
- mean rank: 825.6;
- maximum rank: 9,857;
- top-64 recall: 58.3%;
- top-256 recall: 70.8%;
- top-1,024 recall: 75.0%.

Scientific SHA-256: `8e2bf77aee8648cc7ffa14c9aec22d30e6786da9118714dfa76ba361c027dbee`.

The screen may initialise search but may not discard candidates permanently.

## Joint coordinate solver

Each development trial uses 32 deterministic starting states:

- the top 16 frequency-screened transposition candidates;
- 16 candidates sampled deterministically across the complete block-size/permutation space.

Each starting state undergoes two coordinate cycles:

1. invert the current block permutation;
2. optimise the monoalphabetic key for `50,000` iterations × `5` restarts under the validated train-only trigram-plus-unigram objective;
3. hold the resulting key fixed and enumerate the complete 41,064-candidate transposition space;
4. select the canonical minimal-period highest-scoring transposition.

After the two cycles:

- deduplicate converged transposition states;
- select the highest-scoring state;
- refine its substitution key using `700,000 × 50`;
- re-enumerate the complete transposition space once;
- if the transposition changes, perform one final `700,000 × 50` substitution refinement.

The true key, permutation, block size and plaintext are never used for candidate selection.

## Development gate

Across 24 English development trials:

- mean character recovery at least 70%;
- median at least 90%;
- at least 20/24 trials at least 70%;
- canonical block-size accuracy at least 90%;
- exact canonical permutation accuracy at least 80%.

If this schedule fails, development may vary only seed count and short-cycle budget on development data. It may not use the true candidate rank or test outputs to seed individual trials.
