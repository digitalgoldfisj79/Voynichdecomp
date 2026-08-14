# U5-A — verbose-cipher fresh-key recovery protocol v0.1

Date frozen: 2026-08-14
Status: **FROZEN BEFORE LOCKED U5-A TRIAL GENERATION**
Target status: **VOYNICH SEALED**

## Scientific question

Before any Voynich score is permitted, can a historically hand-executable Naibbe-like verbose family be recovered from fresh hidden keys under a family-known lower-bound condition?

This is deliberately an easier prerequisite than blind recognition. U5-A supplies an oracle for the verbose family's structural role/equivalence classes and asks only whether the hidden plaintext substitution is recoverable. Passing U5-A does **not** identify an unknown text as Naibbe-like and does not open Voynich; it only licenses U5-B blind surface-role/family recognition.

## External family

Forward-family specification: Michael A. Greshko's published Naibbe cipher and reference implementation (`greshko/naibbe-cipher`). The public family normalises Latin-script text by removing diacritics, uppercasing, W→UU, J→I, K→C, and uses a 23-letter normalized alphabet. Plaintext is divided into one- or two-character segments. Singles are rendered through weighted unigram tables; pairs through prefix+suffix tables, with the implementation's unambiguous-compound constraint.

No claim is made that Naibbe is the historical Voynich mechanism. It is a positive-control family.

## Fresh hidden key

The locked hidden key is **one uniform random permutation of the 23 normalized plaintext letters, shared globally across the unigram, prefix and suffix namespaces**. This preserves Naibbe's authentic fact that the same plaintext letter identity underlies its different table roles. Three independent namespace permutations are explicitly disallowed because they would define a harder, non-Naibbe cipher.

For each trial, table/segmentation/surface randomness is independent of the hidden permutation.

## Oracle-role representation

For U5-A only, the solver is supplied the correct family segmentation and table-role equivalence relation. Every emitted unigram or prefix/suffix component is collapsed to its hidden 23-class cipher identity. The resulting sequence is therefore the character stream after the fresh global substitution, while the surface realization remains a member of the verbose family.

This is intentionally favorable. If key recovery fails even here, no harder blind verbose recognition/recovery arm may open under v0.1.

## Language models and source-family disjointness

Two locked test sources are the example source texts distributed with the Naibbe implementation:

- Latin: Pliny, *Naturalis Historia*, Book XVI (`input/examples/nathist_book16.txt`);
- Italian: Dante, *Divina Commedia* (`input/examples/divina_commedia.txt`).

The language models are trained on **different authors and works**:

- Latin training: Caesar, *De Bello Gallico* I–IV, Project Gutenberg #218;
- Italian training: Collodi, *Le avventure di Pinocchio*, Project Gutenberg #52484.

No character from the Dante/Pliny locked test texts is used to fit the trigram or unigram language models.

## Locked trials

- 20 total fresh-key trials: 10 Latin + 10 Italian;
- each trial uses 384 normalized plaintext characters;
- test chunks are deterministic, non-overlapping slices distributed through the normalized locked source;
- a fresh global 23-letter permutation is generated independently for every trial from `SHA256('frontier-u5-a' || language || trial_index)`;
- no U5 locked result is used to select model, schedule, length, source, key or preprocessing.

The 384-character primary length is frozen because the inherited monoalphabetic instrument had already demonstrated near-complete recovery at this length before U5 was conceived. This is an instrument qualification, not a manuscript-length simulation.

## Recovery engine

The solver is the inherited key-invariant `mono_solver_v051.py` simulated-annealing engine from branch `experiment/terminal-cipher-programme-v0.6-20260716`, not a newly tuned U5 solver.

Frozen search schedule: **700,000 iterations × 50 restarts**, the development-selected schedule recorded by the historical V0.5.1 locked monoalphabetic result. No U5-specific schedule search is allowed.

Language model: smoothed character trigram plus the inherited 0.15 unigram term. Initial key: inherited frequency-ranking initialization. Recovery score is normalized Levenshtein character accuracy against the sealed plaintext.

## U5-A gate

The umbrella preregistration is unchanged:

- mean normalized plaintext recovery >= **0.85**;
- at least **16/20** fresh-key trials have recovery >= **0.75**.

U5-A PASS requires both. An early fail is allowed once five trials below 0.75 make 16/20 mathematically impossible; mean recovery is still reported for all trials already run.

### Consequence

- PASS → U5-B blind surface-role/family recognition may be built and calibrated; Voynich remains sealed.
- FAIL → `FAIL_RECOVERY_CALIBRATION`; U5 closes under v0.1; Voynich remains sealed.

## Recognition gate reserved for U5-B

The remaining umbrella requirements are not tested by this easier U5-A arm:

- operational family-recognition recall >=0.80 at precision >=0.95;
- matched-null false-positive rate <=0.05;
- source-family-disjoint recognition test.

Passing recovery without recognition is formally `RECOVERABLE_NOT_IDENTIFIABLE` and does not permit a Voynich score.

## Anti-contamination

No Voynich text, Voynich metric, Voynich token inventory or Voynich section/hand label is read by U5-A. U5-A operates only on known-answer external controls.