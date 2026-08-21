# Stateful Verbose Transducer (SVT) v0.1 — frozen protocol

Date frozen: 2026-08-21
Branch: `experiment/stateful-verbose-transducer-v0.1-20260821`
Parent programme: `experiment/terminal-cipher-programme-v0.6-20260716`
Status: **FROZEN BEFORE NEW SYNTHETIC RESULTS**

## 1. Question

Can a solver recover fresh plaintext from a cipher that combines all three properties below, without being given the segmentation?

1. a changing/stateful substitution alphabet;
2. variable-length 1–3-glyph surface units;
3. hidden or misleading unit boundaries.

Only if that combined mechanism is recoverable on untouched synthetic controls may the identical solver configuration be applied to Voynich.

This is a new mechanism class. It does **not** reopen July Family S by retuning its failed arbitrary-codebook solver. Family S remains a binding negative result for the construction it tested.

## 2. Motivation from prior results

The programme is the intersection left untested by Terminal Cipher v0.6:

- Family P (stateful/poly-alphabetic) passed locked synthetic plaintext recovery strongly.
- Family S (syllabic/polygraphic/unsegmented) failed development recovery and never opened Voynich.
- The blind family-identification bridge later blocked Family P from Voynich.

Subsequent Voynich-only structural work supplies constraints, not plaintext values:

- written spaces are not assumed to be hard unit boundaries;
- cross-boundary last→first dependence is real;
- boundary-spanning bridge objects contain predictive information;
- a simple five-state bridge bottleneck is too small;
- the subsequent five-state × known-production-state rescue also failed its frozen gate, so no five-vowel/product-state factorisation is imported here;
- local production/context state is therefore admissible only as a generic nuisance concept, with no historical language or phonetic assignment.

## 3. Synthetic mechanism

Primary positive family: **factorised stateful verbose transducer (FSVT)**.

A plaintext symbol `x_i` is first encrypted under an **order-free state-specific substitution alphabet**. The state schedule reuses Family-P periodic/line-reset structure, but no circular glyph order is assumed. Each state alphabet is a fresh permutation related to a shared fresh base alphabet by bounded perturbations. This removes the July wheel solver's target-identification defect: EVA does not supply an independently observed ring order. The visible emission for that plaintext symbol is then a 1–3 glyph codeword:

`E_i = h_i r_i1 r_i2`

where zero, one, or two continuation glyphs are emitted. Continuations:

- use the same visible alphabet as heads;
- are conditionally dependent on the head, production state, and a coarse plaintext class;
- therefore do not carry an externally visible "modifier alphabet";
- make segmentation non-trivial while preserving a factorised recoverable mechanism.

Observed line boundaries are preserved. Word/token spaces are absent in the primary synthetic condition. A secondary soft-space condition may insert imperfect spaces, but spaces are never supplied to the decoder as hard boundaries.

Fresh base alphabets, state alphabets, continuation offsets, length distributions, plaintext chunks, and random seeds are disjoint across train/development/locked test.

## 4. Hostile controls

The solver must also see constructions it should **not** claim to solve:

1. `NONFACT`: fresh variable-length codewords whose identity is not reducible to a shared stateful head stream;
2. `SHUFFLED_HEAD`: genuine FSVT surface strings shuffled within observed lines;
3. `GEN`: structured generated text drawn from the existing generator controls where available;
4. `ORDINARY`: ordinary language-like character streams.

The binding v0.1 implementation uses `NONFACT` and `SHUFFLED_HEAD`; `GEN` and `ORDINARY` are reserved independent extensions and cannot be substituted post hoc for a failed binding control.

## 5. Solver architecture

### 5.1 Boundary lattice

No true synthetic boundary is exposed to the joint solver.

Candidate unit starts are proposed from surface statistics only. The primary proposal statistic is transition surprisal: continuation transitions are expected to be relatively predictable, while a new hidden head tends to break that local continuation relation.

A dynamic-programming beam returns the top `B=8` complete segmentations under:

- allowed codeword lengths {1,2,3};
- a weak frozen length prior `(0.30, 0.45, 0.25)`;
- no access to plaintext, key, state schedule, or true codeword lengths.

The same lattice construction is used for synthetic and Voynich data.

### 5.2 Stateful head decoder

Each segmentation induces a candidate hidden-head stream. It is scored by an **order-free Family-P successor**:

- periodic and line-reset state schedules;
- candidate periods 2–12;
- a separate substitution permutation for each candidate state;
- n-gram language-model scoring;
- simulated annealing / key refinement;
- MDL structural penalty for extra state alphabets.

The July Family-P scheduling, scoring discipline and annealing architecture are reused. The modular wheel/ring assumption is retained only as an historical positive control and is not used for Voynich.

### 5.3 Joint score

A candidate receives:

`joint = head_language_score + boundary_score - structure_penalty`

with scale coefficients frozen before locked testing.

No semantic word guessing, manual mapping, or target-specific thresholding is allowed.

### 5.4 Language handling

Synthetic mechanism recoverability is calibrated with the correct source-language model supplied. This isolates whether the combined transducer can actually be inverted. Target application then runs the complete predeclared compatible language-model portfolio automatically. Language ranking is **diagnostic only** unless a separate held-out source-language discriminator passes its own calibration; no prettiest-output/manual language selection is permitted.

## 6. Gates

### Gate 0 — segmentation sanity

Without exposing true boundaries to the candidate generator:

- mean boundary F1 ≥ 0.90;
- median boundary F1 ≥ 0.92;
- at least 18/20 trials ≥ 0.85;
- top-8 lattice contains a path with F1 ≥ 0.90 in at least 18/20 trials.

Failure closes the current surface factorisation before joint decoding.

### Gate 1 — head decoder inheritance

With true segmentation supplied only in this oracle component check, the order-free stateful head solver must retain:

- mean plaintext recovery ≥ 0.90;
- median ≥ 0.95;
- at least 18/20 trials ≥ 0.85;
- structure accuracy ≥ 0.85.

Failure blocks joint testing.

### Gate 2 — joint development, language-oracle

With segmentation hidden:

- mean plaintext recovery ≥ 0.75;
- median ≥ 0.85;
- at least 16/20 trials ≥ 0.70;
- mean boundary F1 ≥ 0.80;
- state/mode structure accuracy ≥ 0.75;
- selected segmentation is not truth-assisted.

One development amendment is permitted, exactly as in Terminal v0.6. The amendment must be frozen before rerun.

### Gate 3 — untouched locked test

Binding gate for target access:

- mean plaintext recovery ≥ 0.75;
- median ≥ 0.85;
- at least 16/20 trials ≥ 0.70;
- mean boundary F1 ≥ 0.80;
- state/mode structure accuracy ≥ 0.75;
- hostile-control false-positive rate ≤ 0.05 at the frozen 0.70 recovery criterion;
- no post-test modification.

Passing this gate licenses a Voynich **mechanism test**, not a source-language claim.

If Gate 3 fails, Voynich remains sealed and the programme closes with a synthetic recoverability failure.

## 7. Voynich transfer

The target runner is physically separate from the synthetic harness and refuses to load Voynich unless a hashed gate JSON states `locked_gate_pass=true` for the frozen configuration.

Target representations:

1. line-aware EVA stream with ordinary spaces erased during inference but their locations retained for later audit;
2. line-aware recurrence-canonical representation where compatible;
3. optional BPE diagnostic representation, never used to tune the primary segmentation.

Voynich evidence requires all of:

- convergence across independent restarts;
- stable segmentation/head assignments across held-out folios;
- stable state structure across manuscript partitions;
- held-out language-model improvement over matched nulls;
- improvement over shuffled-head and generated-text controls;
- auditable character-by-character decoding lattice;
- no manual semantic selection.

If the solver abstains or solutions are unstable, the result is negative/indeterminate, not a decipherment.

## 8. Explicit exclusions

This programme does not test:

- unrestricted nomenclators;
- arbitrary variable-length codebooks with no shared factorisation (already represented by failed Family S constructions);
- unrestricted transposition;
- historical-language-specific values such as Bavarian/MHG mappings;
- five-vowel VBM values;
- manually chosen Voynich segmentation.

## 9. Stopping rule

One implementation and at most one registered development amendment. No locked-test tuning. No target access after a failed locked gate.

A failure means only:

> the tested factorised stateful verbose mechanism is not demonstrably recoverable by the frozen solver at the required reliability.

A pass means only:

> the solver is qualified to test whether Voynich contains evidence for this mechanism.

It does not establish that Voynich is encrypted, German, or meaningful.
