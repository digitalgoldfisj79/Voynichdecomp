# Recoverability frontier v0.5.0 — frozen pilot protocol

Date: 2026-07-15

Status: **development pilot; no Voynich inference permitted**

## Scientific target

Measure how much independently selected plaintext information can be recovered under known synthetic ground truth across heterogeneous languages, cipher mechanisms, text lengths and noise regimes.

This programme does not claim to be a universal cipher-versus-generator arbiter. An explicit `NO_MESSAGE` / abstention output is required for structured controls.

## Corpus boundary

The six pinned Universal Dependencies corpora and their SHA-256 hashes are defined in `corpus_manifest_v050.json`.

- decoder training: corpus `train` partitions only;
- threshold and hyperparameter calibration: `dev` only;
- scientific test outcomes: `test` only;
- deterministic, non-overlapping source chunks;
- no Voynich text.

## Cipher DSL families

1. `mono`: monoalphabetic substitution;
2. `homophonic`: frequency-adaptive homophonic substitution;
3. `null_homophonic`: homophonic substitution with inserted nulls;
4. `polyalphabetic`: periodic additive substitution with unseen periods and keys;
5. `feedback`: stateful plaintext-feedback substitution;
6. `nomenclator`: frequent whole-word codes plus character substitution;
7. `transposition`: substitution followed by block permutation;
8. `fractionated`: coordinate-pair expansion.

Every key and parameter draw is independent by sample. Test keys are never reused in training or development. Test parameter combinations are drawn from ranges offset from development where possible.

## Structured controls

Each control is generated at the plaintext layer and then passed through the exact same cipher family, key-generation, renderer and noise code as a positive:

1. source-trained second-order Markov sequence;
2. motif concatenation and mutation;
3. copied-and-mutated training chunks;
4. slot-based pseudo-word grammar.

Controls have no independently selected held-out plaintext target and are labelled `NO_MESSAGE`.

## Length and noise grid

- target normalized plaintext lengths: 96, 192 and 384 characters;
- channel noise: 0%, 1% and 3%;
- noise includes substitutions, insertions and deletions with deterministic proportions.

## Evaluation arms

### A. Channel oracle

The true key, segmentation and family are supplied to the inverse transform. This measures information destroyed by the channel and checks every cipher implementation for round-trip correctness.

### B. Family-known decoder

The decoder receives ciphertext, source-language tag and cipher-family tag, but no key, plaintext, segmentation map or generator label.

### C. Blind-family decoder

The decoder receives ciphertext and source-language tag only. It must infer whether a message is present and recover it without knowing the cipher family.

Both learned arms use fresh random initialisation and synthetic training only. No pretrained language model is permitted in v0.5.0 because the public test corpora may occur in pretraining data.

## Model and compute

A compact sequence-to-sequence Transformer is trained from scratch with a separate message-presence classifier. Positive samples contribute classification and plaintext reconstruction loss; controls contribute classification loss only.

Family-known and blind-family models are independent jobs and may run in parallel. GPU compute is used for training; corpus acquisition, generation and oracle checks are CPU-parallel.

## Primary metrics

- oracle normalized character accuracy;
- learned normalized character accuracy on positives declared message-bearing;
- exact sequence recovery;
- positive message-detection sensitivity;
- control false-positive rate;
- family-wise worst-case recovery;
- retention from family-known to blind-family decoding;
- calibration and abstention rate.

Character accuracy is one minus normalized Levenshtein distance, floored at zero.

## Development sample size

Default full pilot per learned arm:

- training: 120,000 positives and 120,000 controls;
- development: 8,640 positives and 8,640 controls;
- test: 8,640 positives and 8,640 controls.

The test grid corresponds to 6 languages × 8 cipher families × 3 lengths × 3 noise levels × 20 independent samples.

## Gates

### Channel gate

- noiseless oracle character accuracy at least 99.9% for every family;
- 1% and 3% degradation reported, not threshold-tuned;
- any non-invertible implementation defect stops learned training.

### Family-known go gate

- message sensitivity at least 80%;
- control false-positive rate at most 5%;
- mean character accuracy at least 70% in at least five materially distinct cipher families;
- no claimed success based only on language fluency or source identification.

### Blind-family go gate

- retains at least 85% of family-known mean character accuracy;
- control false-positive rate at most 5%;
- no family below 50% character accuracy if an overall general-decoder claim is made.

### Stop rules

Stop before broad multilingual expansion if:

- channel oracle fails;
- controls trigger above 5% after development calibration;
- recovery disappears on unseen keys or parameter combinations;
- apparent success is driven by one language or one cipher family;
- threshold or model selection is modified after test inspection.

## Scientific boundary

Passing v0.5.0 establishes a bounded synthetic recoverability frontier only. It does not prove that the Voynich Manuscript is a cipher or generated text. A later real-text programme must use bounded model comparison and permit `NON_IDENTIFIABLE` as an outcome.
