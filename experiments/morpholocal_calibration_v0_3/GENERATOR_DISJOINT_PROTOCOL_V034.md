# Morpholocal calibration v0.3.4: generator-disjoint protocol

Status: development protocol, frozen before execution.

## Question

Does the v0.3.3 latent-order randomization statistic distinguish independently generated enciphered text from equally ordered hidden-state processes that contain no plaintext message?

## Fixed detector

- v0.3.3 scientific base: `09cc07eb0915f7d956ba4cbf522b9b08e8758261`.
- Solver: beam only.
- Sequence test: within-line permutation with the first latent unit fixed.
- Randomizations: 199.
- Decision: one-sided `p <= 0.05` and positive transition-codelength advantage.
- No threshold or feature changes after results are observed.

## Generator-disjoint positives

Source words are taken in original order from repository corpora not used to fit the detector's Currier-I external model:

1. `Paper/Cipher_paper/greek_corpus_parsed.pkl`
2. `Paper/Cipher_paper/greek_dmm_corpus.pkl`

Surface ciphertext is produced by independently written mechanisms rather than v0.2 `generate_cipher_trial`:

1. keyed-PRF homophony, global key, no nulls;
2. rotor homophony, Currier-split keys, no nulls;
3. feedback homophony, global key, two null cells;
4. line-keyed homophony, Currier-split keys, two null cells.

The independent renderer uses the frozen 24-cell surface registry only as an input alphabet. It does not sample the detector's external transition matrix and does not call its four emission-policy generators.

## Ordered non-message controls

Controls use the same surface renderer and key-size constraints but their latent sequences are produced without a plaintext corpus:

1. random strongly ordered hidden Markov process;
2. finite motif grammar;
3. document-topic finite-state process;
4. copied-line process with mutation.

These controls are deliberately harder than the v0.3.3 controls because they contain genuine latent order while carrying no source message.

## Development smoke

- 16 positives: four cipher mechanisms x two corpora x two deterministic replicates.
- 16 controls: four ordered-control families x four deterministic replicates.
- 12 documents per trial, 180 surface tokens per document.
- All seeds, corpus hashes, source hashes, trial records and summaries must be written to the result artifact.

Escalation requires all of:

- sensitivity at least 65%;
- false-positive rate at most 20%;
- at least three of four positive mechanisms detected in at least half their cases;
- no single ordered-control family accepted in more than half its cases.

Failure stops the programme for comparator redesign. Passing permits a separately frozen large locked validation. The smoke result itself is not admissible as a Voynich inference.
