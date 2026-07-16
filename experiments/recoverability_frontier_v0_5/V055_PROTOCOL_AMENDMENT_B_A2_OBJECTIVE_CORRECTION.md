# Recoverability frontier v0.5.5 — protocol amendment B: A2 objective correction

Date: 2026-07-16

Status: fixed after the first full Stage A execution, before the valid A2 rerun.

## Implementation defect

The first full Stage A job bound `mono_solver_v051_search2` to the homophonic programme's character quadgram scorer. This was done to resolve a tensor-rank compilation error, but it contradicted the frozen A2 specification, which required the **passing v0.5.1 monoalphabetic solver**.

The validated v0.5.1 solver uses its own smoothed character trigram plus unigram objective and a search schedule calibrated to that score scale. Under the wrongly bound quadgram objective, the search systematically moved away from strong frequency initialisations and produced 14.84% mean recovery.

That A2 output is an implementation-invalid diagnostic and must not be interpreted as evidence against recoverability.

## Valid correction

- Retain all generated ciphertexts, source chunks, keys, permutations and oracle information unchanged.
- Retain A1 results; A1 uses a separate exact quadgram enumeration routine and is unaffected.
- Rerun A2 only.
- Use `mono_solver_v051.build_language_model` and the unmodified `mono_solver_v051_search2.anneal_mono_search2`.
- Retain the frozen `700,000` iterations × `50` restarts schedule.
- Retain the same eight development replicates for block sizes 4, 6 and 8.

The valid A2 run replaces, rather than supplements, the invalid A2 output from job `Digitalgoldfish79/6a5869afb1669a49bf076cc5`.
