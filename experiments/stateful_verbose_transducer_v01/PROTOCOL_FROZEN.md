# Stateful Variable-length Transducer (SVT) v0.1 — frozen protocol

Frozen: 2026-08-21
Branch: `experiment/stateful-verbose-transducer-v0.1-20260821`

## Question

Can a solver recover ordinary plaintext from a cipher that combines **hidden variable-length code units** with **state-dependent arbitrary substitutions**, without being given code-unit boundaries, the substitution dictionaries, or the state schedule; and, only if that succeeds on fresh locked synthetic data, does the same frozen mechanism provide transferable evidence on Voynich?

This is a new programme. It does not reopen or retune Terminal Cipher v0.6 Family S. Family S failed because hidden segmentation/generalisation failed. It reuses only the pinned corpora, language models, scoring infrastructure, and the successful principle from Family P that state/schedule hypotheses must be inferred and scored rather than supplied.

## Mechanism family

Plaintext symbols `p_j` are generated sequentially. A cipher state `s_j` is determined by a bounded schedule. For each state, a fresh arbitrary codebook maps plaintext symbols to opaque surface strings of length 1–3:

`p_j --(state s_j, fresh codebook)--> c[i:k]`, where `1 <= k-i <= 3`.

The codewords are concatenated without separators. The solver therefore does **not** know how many ciphertext glyphs correspond to one plaintext unit. Codebooks are independently random across synthetic trials and may reuse identical surface strings across different states with different plaintext values. Within a state, codewords are injective.

No circular ordering of visible symbols exists or is supplied. This explicitly removes the invalid Voynich-facing assumption in the successful July Family-P wheel solver.

### State schedules

Primary synthetic schedules:

1. `periodic`: state = decoded plaintext-unit index modulo period;
2. `line_reset`: same schedule but phase resets at independently observed line boundaries.

Development periods: 2, 3, 4.
Locked-test periods: 5, 6, plus fresh instances at 2–4.

The state index is tied to **decoded unit position**, not ciphertext-glyph position. A variable-length renderer therefore cannot leak state phase.

## Solver

The joint solver is an order-free beam transducer. Each live hypothesis contains:

- current ciphertext position;
- decoded plaintext-unit count;
- previous two plaintext symbols for language-model scoring;
- candidate state schedule;
- per-state map from observed 1–3-glyph codeword to plaintext symbol;
- inverse per-state map enforcing injectivity;
- recovered plaintext;
- recovered code-unit boundaries.

At each expansion it considers ciphertext substrings of length 1, 2 and 3. If a substring is already mapped in the current state, its plaintext value is fixed. If new, the solver branches over high-probability plaintext symbols not already assigned in that state. Ranking uses only a train-split character trigram language model, a fixed code-length prior, and a fixed MDL charge for introducing a new mapping.

Candidate periods/modes are solved separately and selected by the same penalised score. Language-blind runs solve every admitted language independently and select by normalised held-out score plus the frozen structural penalty.

## Calibration sequence

Voynich is inaccessible to the synthetic runner and no target file is imported by it.

### Stage A — oracle boundaries

True code-unit boundaries are supplied; codebooks, plaintext and state dictionaries remain hidden. Purpose: establish that the stateful substitution component is recoverable when segmentation is not the obstacle.

Gate per language-length cell:

- mean plaintext recovery >= 0.90;
- median >= 0.95;
- at least 7/8 trials >= 0.80.

Failure blocks later stages and is diagnostic of the mapping/state solver.

### Stage B — hidden boundaries, oracle schedule family/period

Boundaries are hidden. True schedule family and period are supplied; codebooks and plaintext remain hidden.

Gate:

- mean plaintext recovery >= 0.80;
- median >= 0.90;
- mean boundary F1 >= 0.85;
- at least 7/8 trials >= 0.70 plaintext recovery.

Failure means segmentation remains unsolved and blocks joint testing.

### Stage C — fully joint development

Hidden boundaries, hidden period, hidden periodic-vs-line-reset mode, hidden codebooks. Language identity is supplied only in the first arm.

Gate per admitted language-length cell:

- mean plaintext recovery >= 0.75;
- median >= 0.90;
- at least 7/8 trials >= 0.70;
- mean boundary F1 >= 0.80;
- mode accuracy >= 7/8;
- period accuracy >= 6/8.

One development amendment is permitted only if a failure is attributable to a solver defect visible on synthetic truth. The amendment must be committed before any locked test.

### Stage D — language-blind development

The correct language is not supplied. Candidate language models are the pinned v0.5 corpus set. In addition to Stage-C recovery conditions:

- correct language selected >= 7/8;
- wrong-language solutions must not achieve comparable plaintext recovery under truth scoring;
- the selected score margin must be positive in >= 7/8.

### Stage E — locked synthetic test

Fresh plaintext chunks, fresh codebooks, fresh periods including 5 and 6, fresh random seeds. No tuning after this point.

Required aggregate conditions:

- mean plaintext recovery >= 0.75;
- median >= 0.90;
- >= 14/16 trials >= 0.70;
- mean boundary F1 >= 0.80;
- correct mode >= 14/16;
- correct period >= 12/16;
- correct language >= 14/16 in language-blind arm.

Only an immutable `LOCKED_PASS.json` produced by this stage may authorise construction or execution of a Voynich target runner.

## Hostile controls

The synthetic suite includes:

- ordinary monoalphabetic substitution with no verbose layer;
- variable-length code without state changes;
- stateful substitution with fixed length 1;
- random/motif structured text without plaintext;
- codeword-length distribution shifts between development and locked test.

A positive classifier/decoder must not hallucinate variable-length state structure on the fixed-length controls merely because a higher-complexity model can improve training likelihood.

## Complexity discipline

The following are frozen before development:

- maximum code length = 3;
- schedule period candidate set = 2–6;
- one-to-one plaintext assignment within each state;
- beam width and new-mapping penalty as recorded in `CONFIG_FROZEN.json`;
- code-length prior as recorded in `CONFIG_FROZEN.json`;
- trigram language model training only on the pinned train split;
- no semantic word guessing;
- no manual boundary insertion;
- no target-language-specific changes after locked test.

## Voynich consequence

A failed synthetic gate says only that the instrument cannot test this family. It is not evidence against Voynich being verbose/stateful.

If all locked gates pass, the Voynich application must still satisfy:

1. the same state/segmentation parameters transfer across held-out folios;
2. independent restarts converge on equivalent latent structure;
3. decoded-unit boundaries are stable to transcription representation;
4. a single global decoder improves held-out language-model likelihood;
5. long decoded plaintext units do not show the wrong-key collapse seen in prior substitution-class tests;
6. matched structured-generator controls do not produce equivalent evidence;
7. no output-selected language/key interpretation.

No Voynich plaintext may be inspected under v0.1 unless Stage E passes.