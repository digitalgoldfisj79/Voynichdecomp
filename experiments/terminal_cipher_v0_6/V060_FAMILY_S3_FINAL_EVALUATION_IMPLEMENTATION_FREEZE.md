# v0.6 Family S3 — final neural ensemble evaluation implementation freeze

Date: 2026-07-16

Status: **FROZEN BEFORE LOADING EITHER CHECKPOINT AGAINST DEVELOPMENT TRIALS.**

This document completes operational details required to execute the already-registered final S3 amendment. It changes no architecture, training data, model seed, update count, solver budget, gate or split. No development recovery, test data, Voynich data or Davis labels were inspected in fixing these details.

## Inputs

- models: seed `1731` and seed `1732`, final update `30000` only;
- synthetic language: frozen English corpus and inventory already used by S1/S2/S3;
- split: `dev` only;
- length: 384 plaintext characters;
- replicates: 16;
- checkpoints must be reconstructed from their three stored parts and match the manifest SHA-256 before inference.

The original S1 development records contain no internal line metadata. They are therefore represented as one observed line: source position zero has line-start flag one and every later source position has flag zero. No hidden plaintext or true boundary is used to construct line flags.

## Ensemble inference

For each model, the ciphertext is first-occurrence canonicalised exactly as in training. Encoder outputs are computed independently.

- plaintext posterior: arithmetic mean of the two models' softmax probabilities at each autoregressive step;
- direct decoding: beam width 4, fixed output length 384, cumulative log probability, deterministic lexicographic tie-breaking;
- boundary posterior: arithmetic mean of the two sigmoid boundary probabilities.

## Boundary lattice

The eight highest-scoring complete segmentations are obtained by exact k-best dynamic programming over code lengths 1, 2 and 3.

For a segment from source positions `left` through `right`, inclusive, its score is:

- `log P(length)` under the frozen prior `{1: 0.20, 2: 0.45, 3: 0.35}`;
- `log(1 - boundary_probability)` for every internal source position;
- `log(boundary_probability)` at the segment end.

Probabilities are clipped to `[1e-7, 1 - 1e-7]`. The complete path must end at the final source symbol. Ties are ordered lexicographically by the tuple of segment lengths. Boundary F1 is evaluated on the highest posterior path, independently of plaintext selection.

## Lattice mapping and refinement

Each of the eight segmentations is passed to the unchanged S2 unit-language mapping solver at:

- 700,000 iterations;
- 50 restarts;
- the existing stable label-derived seed.

The candidate with the highest existing S3 `combined_score` is selected. That same segmentation is then rerun at:

- 700,000 iterations;
- 200 restarts.

No true segmentation, codebook, plaintext or recovery score participates in selection.

## Direct-versus-lattice calibration

The registered amendment requires selection using calibrated neural-model and unit-language likelihoods learned solely from synthetic train examples. This is implemented before development scoring as follows.

Calibration data:

- 32 fresh examples from `SyntheticGenerator` using the frozen train stream;
- deterministic generator seed `stable_seed("v060-s3-selection-calibration", 1731, 1732)`;
- each true target is one positive candidate;
- six deterministic negative candidates per example: symbol replacements at 5%, 10%, 20% and 35%, plus deletions at 5% and 10%;
- corruption seeds are derived only from the calibration example index and corruption label.

Candidate features:

1. ensemble teacher-forced mean log probability per scored character, conditioned on that example's ciphertext;
2. frozen S2 unit-language score per plaintext character after deterministic longest-first unitisation;
3. negative absolute output-length deviation from 384, divided by 384.

For candidates longer than 384, neural likelihood is evaluated on the first 384 characters; the separate length feature retains the full deviation. Empty candidates receive a finite floor score.

The three features are standardised using the complete calibration set. A class-balanced logistic regression is fitted by exactly 1,000 full-batch gradient steps, learning rate 0.05 and L2 penalty 0.001, with zero initial weights and intercept. The fitted logit is the frozen candidate-selection score.

For each development trial, the direct beam candidate and the fully refined lattice candidate are scored by this fitted classifier. The higher logit is selected; exact ties select the direct candidate. Development truth is used only after selection to calculate the registered metrics.

## Registered gate

The final S3 development gate remains unchanged:

- mean plaintext recovery at least 75%;
- median plaintext recovery at least 85%;
- at least 13 of 16 trials at or above 75%;
- mean boundary F1 at least 85%;
- no trial below 40% plaintext recovery.

A failure closes Family S. A pass freezes this implementation and permits one untouched locked-test run. No post-result modification is permitted.