# Recoverability frontier v0.5.5 — substitution plus block transposition

Date frozen: 2026-07-16

Status: **component-oracle development programme; no Voynich inference permitted**

## Motivation

v0.5.4 nomenclator Stage B is blocked because its component gates did not both pass. The programme therefore moves to a structurally different cipher family rather than modifying failed gates.

v0.5.5 studies a fresh monoalphabetic substitution followed by a fixed permutation within repeated blocks.

## Generator correction

The v0.5.0 transposition generator appended a dedicated padding symbol whenever plaintext length was not divisible by the block size. That symbol and its tail location could reveal both block alignment and effective message length.

v0.5.5 removes that shortcut:

1. select a source chunk whose scored length is an exact multiple of the block size;
2. apply a fresh monoalphabetic character substitution;
3. apply a fresh non-identity permutation to every complete block;
4. apply a fresh joint relabelling of substituted surface symbols;
5. canonicalise symbols by first occurrence;
6. add no padding or boundary markers.

Spaces are ordinary plaintext characters and are transposed with all other symbols. Word boundaries are therefore not externally visible.

## Corpus and conditions

- existing six hash-pinned corpora and train/dev/test partitions;
- initial language: English;
- primary scored length: 384 normalized characters;
- block sizes: 4, 6 and 8, all exactly dividing 384;
- eight development chunks and twenty future locked-test chunks per block size;
- fresh substitution key and block permutation per trial;
- no channel noise.

A later extrapolation set may use block sizes 5 and 7 with separately divisible scored lengths. It is not part of initial selection.

## Stage A — component oracle gates

### A1. True substitution key; hidden block permutation

Supplied:

- true mapping from canonical cipher symbols to substituted plaintext characters;
- block alignment begins at sequence position zero;
- candidate block-size set `{4, 6, 8}`.

Hidden:

- true block size;
- within-block permutation.

For every candidate block size, enumerate every permutation, invert the transposition, and score the resulting plaintext with the train-only character quadgram model. The selected candidate is the highest-scoring complete sequence.

Metrics:

- block-size accuracy;
- exact permutation accuracy;
- normalized character recovery;
- rank and score margin of the true candidate.

### A2. True block permutation; hidden substitution key

Supplied:

- true block size and permutation;
- true observed plaintext-character inventory, but not its assignment.

Hidden:

- monoalphabetic substitution mapping.

The solver first exactly inverts the block permutation, then applies the passing v0.5.1 key-invariant monoalphabetic solver under the train-only language objective.

Metrics:

- substitution-key accuracy;
- normalized character recovery;
- objective margin over frequency matching.

## Stage B — joint family-known recovery

Stage B is prohibited unless both component gates pass.

Candidate strategy:

1. enumerate block size and permutation candidates;
2. derive recurrence-canonical detransposed sequences;
3. run a bounded monoalphabetic search for each candidate;
4. rank complete candidates by a common train-only objective with an explicit search-cost penalty;
5. retain an abstention state when candidate evidence is diffuse.

Development may prune candidates using substitution-invariant recurrence statistics, but no true key, true permutation or test plaintext may be used.

## Stage A development gates

For each of block sizes 4, 6 and 8:

- A1 mean character recovery at least 99%;
- A1 block-size accuracy 8/8;
- A1 exact permutation accuracy at least 7/8;
- A2 mean character recovery at least 90%;
- A2 median recovery at least 99%;
- A2 at least 7/8 trials recover at least 90%.

Both arms must pass every block-size condition before Stage B.

## Stage B development gate

Across 24 English development trials:

- mean recovery at least 70%;
- median at least 90%;
- at least 20/24 trials recover at least 70%;
- block-size accuracy at least 90%;
- exact permutation accuracy at least 80%;
- false confident selection on matched structured-generator controls below 5%.

## Locked English test gate

A frozen Stage B solver is evaluated once on sixty untouched English trials:

- mean recovery at least 70%;
- median at least 90%;
- at least 48/60 trials recover at least 70%;
- block-size accuracy at least 90%;
- no post-test changes.

## Boundary

Passing establishes recoverability only for repeated fixed block permutations composed with monoalphabetic substitution. It does not cover arbitrary columnar transposition, route ciphers, variable blocks, nulls, fractionation or manuscript layout operations. No Voynich data may be scored before locked synthetic validation.