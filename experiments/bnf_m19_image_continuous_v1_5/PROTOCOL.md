# BnF M19 Image Bridge v1.5 — Continuous-Emission Protocol

Date: 2026-08-09
Parent: v1.4 dense-segmental image-gate negative.
HF source revision: `Digitalgoldfish79/vdino3-crops@ea597db8ff2c06631c4c311d90c8cf0418f5e26c`.

## Motivation

v1.4 gives a specific image-only result: dense DINO segment boundaries are reproducible (best Tvis boundary F1 0.8864 at lambda=.12), while 19 hard visual class labels are not (best class stability 0.6890). v1.5 therefore preserves the prospectively selected visual boundary model but removes the unsupported discrete-glyph assumption.

The hypothesis tested is now:

`plaintext language -> BnF M19 numerical state -> continuous image-segment appearance`

Each of the 19 numerical states has a continuous dense-DINO emission distribution. No EVA/transliteration label enters fitting or scoring.

## Sealing and data split

All textual/transliteration fields remain sealed: `word`, `eva_aligned`, `eva_glyph`, `word_len` and related strings are forbidden until terminal audit.

Use the exact v1.2 split: T12=112 folios, H12=45, C12=68, Tfit/Tvis=90/22. C12 is inaccessible to language scoring unless H12 passes.

Image segmentation is frozen to v1.4 raw dense DINO, K=19 visual segmenter, lambda=.12, adjacent component groups length 1-3, exactly three visual fit/segment alternations. Fit the boundary model on all T12 only; use the frozen visual centroids/lambda to segment H12 and, only if unlocked, C12. The segment-cluster labels are discarded. The continuous segment embedding is the normalized mean of its member dense component vectors.

## Continuous M19 model

For language L, the language model induces the same 19-state numerical Markov prior used in v0.7-v1.4 (`induced(L)`): start, transition and end probabilities over BnF numerical values.

Observed segment vector x in R^768 is emitted from numerical state v by a spherical Gaussian with state mean mu_v and a single shared sigma fixed independently of language.

### Sigma calibration — image only

Before any language fit:
1. fit unsupervised K=19 MiniBatchKMeans to a deterministic <=80k Tfit segment sample;
2. calculate residual RMS per coordinate `sigma0 = sqrt(mean(||x-centroid||^2)/768)`;
3. evaluate scale factors {0.75,1.0,1.5,2.0,3.0} on Tvis under the fixed KMeans Gaussian mixture, including the Gaussian normalization constant;
4. choose the scale with highest Tvis mixture log-likelihood. Ties within 0.001 nats/segment choose the larger sigma (more conservative overlap).

This fixes sigma before any language-dependent fitting.

### Language-specific fit

For each candidate language independently:
1. initialize 19 means from two independent unsupervised KMeans fits (seeds 408 and 409);
2. obtain an initial cluster->M19 permutation by optimizing the corresponding hard T12 cluster sequence under that language's induced M19 prior;
3. convert cluster centroids to numerical-state means under that permutation;
4. run exactly 8 Baum-Welch/EM iterations on T12 with transition/start/end priors fixed and sigma fixed; only the 19 means are updated;
5. repeat for both visual initializations. Select the higher-T12-likelihood fit for scoring, but retain both for reproducibility.

Mean-fit reproducibility is the posterior-state-frequency-weighted mean cosine between corresponding numerical-state means from the two independent fits. No Hungarian relabelling is allowed because M19 state identities are fixed by the BnF values.

### Scores

For a held-out panel and fitted language model calculate:
- `joint`: exact 19-state HMM log likelihood per segment with continuous emissions;
- `length_only`: the same language's start/transition/end model marginalized over unobserved states for each observed word length;
- `visual_gain = joint - length_only - gaussian_constant`, per segment.

Primary ranking is by `visual_gain`; raw joint ranking is reported as a concordance diagnostic.

## Instrument qualification

Before H12 may be scored, qualify the continuous solver on six fresh controls: Latin, Italian, German, French, Arabic and Spanish.

For each control:
- take 84,000 held-out corpus letters, split 45,000 train / 39,000 hold;
- generate exact M19 numerical states from plaintext letters by uniformly choosing among each letter's distinct BnF values;
- assign 19 random unit prototype vectors in 64 dimensions;
- generate each observed vector as `normalize(prototype_v + epsilon)`, with epsilon calibrated so the total expected noise norm matches the real-image sigma selected above after dimension scaling;
- fit each of the eight candidate language models exactly as for Voynich (two independent visual initializations, 8 EM iterations, fixed sigma appropriate to the synthetic dimension).

Qualification gate:
- correct language ranks first by visual_gain in 6/6 controls;
- minimum visual_gain margin >=0.03 nats/segment;
- correct-language numerical-state mean recovery >=0.90 median cosine and >=0.80 minimum across controls after direct state identity comparison;
- minimum independent-fit reproducibility >=0.90.

If any condition fails: `CONTINUOUS IMAGE INSTRUMENT NOT QUALIFIED`; no H12 score.

## H12 primary gate

Fit all eight candidate languages using T12 only and score H12. A primary candidate requires:
- same language rank 1 by visual_gain and raw joint score;
- visual_gain margin to runner-up >=0.03 nats/segment;
- raw joint margin to runner-up >=0.03 nats/segment;
- independent-fit mean reproducibility >=0.90;
- all 19 states have >=100 posterior-effective T12 assignments;
- H12 segmentation coverage >=0.90.

If this fails, do not score C12.

## C12 confirmation

If H12 passes, freeze the winning T12 means and all visual/segmentation parameters verbatim. Score C12 without refitting.

Confirmation requires:
- candidate rank 1 by both visual_gain and joint;
- visual_gain margin >=0.03 and joint margin >=0.03;
- coverage >=0.90;
- candidate visual_gain exceeds every other language in all four frozen C12 folio buckets.

## Post-confirmation falsification

Only after C12 confirmation:
1. within-word segment-order shuffle, 200 nulls; candidate observed gain margin >99th percentile;
2. BnF-incidence null: permute the 23 complete five-value profiles among plaintext letters, rebuild induced priors and refit candidate on T12 for 200 deterministic permutations; observed C12 candidate margin >99th percentile;
3. word-boundary removal diagnostic;
4. independent feature replication using CLS segment embeddings with the same fixed boundaries and no boundary refit; candidate remains rank 1 on C12 with visual_gain margin >=0.02.

EVA audit fields remain sealed until these falsifications finish.

## Verdicts

- `CONTINUOUS IMAGE INSTRUMENT NOT QUALIFIED`
- `NO CONTINUOUS IMAGE-M19 SIGNAL`
- `H12 CONTINUOUS IMAGE-M19 CANDIDATE / C12 FAILED`
- `CONFIRMED CONTINUOUS IMAGE-M19 SIGNAL <language>`
- `CONFIRMED BUT NULL-SENSITIVE`

No plaintext or dialect claim follows from a statistical signal without independently readable recovery.
