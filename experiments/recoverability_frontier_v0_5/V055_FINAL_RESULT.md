# Recoverability frontier v0.5.5 — final result

Date: 2026-07-16

Verdict: **COMPONENTS RECOVERABLE; JOINT UNKNOWN-BLOCK-SIZE SOLVER FAILS THE FINAL FROZEN DEVELOPMENT GATE. NO LOCKED TEST IS PERMITTED. V0.5.5 IS CLOSED.**

No Voynich text was scored.

## Family

The synthetic family combines:

- a fresh monoalphabetic substitution key;
- a fresh repeated block permutation of canonical size 4, 6 or 8;
- a fresh joint surface-symbol relabelling;
- first-occurrence recurrence canonicalisation;
- no padding, block markers, exposed word boundaries or true block-size information.

The scored plaintext length is exactly 384 characters.

## Component gates

### A1 — true substitution key, hidden transposition

Across 24 English development trials:

- mean plaintext recovery: 100%;
- canonical block-size recovery: 24/24;
- canonical exact permutation recovery: 24/24;
- true candidate rank: 1 in every trial.

A1 passed.

### A2 — true transposition, hidden substitution

Using the validated v0.5.1 monoalphabetic solver (`700,000 × 50`):

- mean recovery: 99.7070%;
- median: 100%;
- minimum: 98.1771%;
- all 24 trials exceeded 98%;
- mean key accuracy: 97.6496%.

A2 passed.

## Initial joint coordinate search

Scientific SHA-256: `109a088738be8eb9996dd116f84208d4371e527743dec597be6bbe8dfa0e7750`

Results:

- mean recovery: 57.9753%;
- median: 32.5521%;
- trials at least 70%: 11/24;
- block-size accuracy: 75.0%;
- exact permutation accuracy: 45.8333%.

This failed development.

## Final permitted stratified development search

Job: `Digitalgoldfish79/6a587047b1669a49bf076d78`

Scientific SHA-256: `ada318018f80d58173ed661b16ae514b3412fcbec202bbdc6279d40453927a20`

Frozen final schedule:

- 16 frequency-ranked starts within each candidate block-size family;
- 16 deterministic random starts within each family;
- up to 96 intended starts, with duplicates removed deterministically;
- three coordinate cycles;
- `100,000 × 8` mono search per cycle;
- complete enumeration of all 41,064 transposition candidates after each cycle;
- final `700,000 × 50` mono refinement;
- one complete final transposition re-enumeration;
- unchanged language model, corpus, ciphertexts and gates.

### Aggregate results

- mean recovery: **76.5299%**;
- median recovery: **99.4792%**;
- minimum recovery: **16.9271%**;
- trials at least 70%: **17/24 (70.8333%)**;
- trials at least 90%: **17/24 (70.8333%)**;
- exact plaintext recovery: **6/24 (25.0%)**;
- canonical block-size accuracy: **23/24 (95.8333%)**;
- exact permutation accuracy: **17/24 (70.8333%)**.

### By true block size

| Block size | Mean recovery | Median recovery | Trials ≥70% | Block-size accuracy | Exact permutation |
|---:|---:|---:|---:|---:|---:|
| 4 | 89.5182% | 99.6094% | 7/8 | 8/8 | 7/8 |
| 6 | 80.1758% | 99.4792% | 6/8 | 7/8 | 6/8 |
| 8 | 59.8958% | 60.8073% | 4/8 | 8/8 | 4/8 |

### Frozen development gates

- mean recovery at least 70%: **pass**;
- median recovery at least 90%: **pass**;
- at least 20/24 trials at least 70%: **fail (17/24)**;
- block-size accuracy at least 90%: **pass (23/24)**;
- exact permutation accuracy at least 80%: **fail (17/24)**.

Overall gate: **fail**.

## Oracle-block-size diagnostics

Three additional runs fixed the candidate block size and are explicitly diagnostic rather than registered Stage B results:

- size 4: 7/8 trials at least 70%, 87.5% exact permutation;
- size 6: 5/8 trials at least 70%, 62.5% exact permutation;
- size 8: 4/8 trials at least 70%, 50.0% exact permutation.

These diagnostics confirm that failure is not primarily block-size classification. It remains a bimodal joint-basin problem even when the correct family dimension is supplied.

## Interpretation

The class is identifiable under either oracle component, and successful joint trajectories normally achieve approximately 98–100% recovery. However, the full blind solver cannot reach the correct joint basin reliably enough on unseen fresh keys, despite exhaustive transposition enumeration, stratified starts and substantial monoalphabetic search.

This is a solver-reliability failure, not evidence that the synthetic family itself lacks a plaintext. It nevertheless prevents defensible blind application: a failed output could not be distinguished from a search failure, and selective retention of attractive outputs would be invalid.

## Decision

- No locked test is opened.
- No further seed, cycle, objective, length or threshold modifications are permitted within v0.5.5.
- Fixed block transposition is closed under the tested blind-recovery protocol.
- The generic recoverability frontier now contains completed work on monoalphabetic substitution, fresh-key homophony, nomenclators and substitution-plus-block-transposition.
- The next phase must be finite and terminal: cover only material historical primitives not already represented, use synthetic-recovery gates before any Voynich application, preserve explicit abstention, and stop rather than tune after test failure.