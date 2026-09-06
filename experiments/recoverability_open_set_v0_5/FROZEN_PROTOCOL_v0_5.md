# Frozen protocol — recoverability-first open-set programme v0.5

**Frozen:** 2026-07-25, before the v0.5 formal outputs were generated.  
**Branch:** `experiment/voynich-recoverability-open-set-v0.5-20260725`  
**Upstream:** v0.4 formal verdict `ABSTAIN_OOD` remains unchanged.  
**Scope:** generability → recoverability → recognisability → transferability. No glyph receives a semantic translation.

## 1. Scientific question

Does the Voynich token stream contain recoverable, transferable operational variables characteristic of human technical notation, as distinct from ordinary language, simple cipher, surface nulls and unconstrained structured generation?

The programme does not ask whether Voynich resembles one named notation family. It tests whether architecture can be recovered and transferred under identity-neutral representations, and permits an open-set `ABSTAIN` result.

## 2. Workstreams

### A. Representation-leakage audit

Re-run grouped historical-family recognition under identity-neutral surface maps:

1. frequency-rank canonicalisation;
2. within-event equality pattern;
3. length plus equality pattern;
4. run-length/equality signature;
5. event inventory rank sequence;
6. character-class-neutral sequence features;
7. sequence/entropy-only features.

The group is the Ammerbach book or GABC source manuscript. No stave, event or transformed derivative of a manuscript may cross a train/test boundary.

**Gate A:** historical notation ROC AUC must be at least 0.80 under at least three identity-neutral maps, and at least one map omitting explicit event-length features must pass. Otherwise downstream family interpretation is disabled.

### B. Known-field recoverability

#### B1. Ammerbach paired-channel alignment

Use the two canonical annotation rows—duration/special and pitch/rest—as two observed channels. Train pointwise association and local-context models on one book and test on the other. For each held-out stave rank offsets -6…+6; the true alignment is offset zero.

Report top-1 alignment recovery, mean reciprocal rank and true-vs-best-false score margin. Repeat after independent symbol permutation within each book.

**Gate B1:** zero offset must rank first for at least 70% of held-out staves in each cross-book direction and after symbol permutation.

#### B2. Neume boundary recovery

Concatenate GABC notation events without boundaries. Train a character-boundary model on whole manuscripts and test on held-out manuscripts. Inputs are local equality-pattern and frequency-rank context only; literal character identity is excluded. Compare against a renewal baseline using only the training event-length distribution.

Report boundary precision, recall, F1, exact-event recovery and improvement over baseline separately for Aquitanian and square notation.

**Gate B2:** grouped boundary F1 must exceed 0.70 and the renewal baseline by at least 0.10 in each historical family.

### C. Voynich compositionality and operational-variable tests

Use complete folios as holdout units and analyse both lossless P70 and the simpler no-suffix decomposition.

1. **Frame substitution:** compare context-distribution similarity for tokens sharing a frame but changing core, versus matched token pairs sharing length/frequency only.
2. **Core substitution:** compare tokens sharing a core but changing control frame.
3. **Neighbour effect:** test whether frame changes predict different next-prefix/line-position distributions after matching on core.
4. **Cross-section transfer:** learn contextual equivalence classes in all but one section and score the held-out section.
5. **Order control:** repeat after within-line token shuffling and after conditional resampling preserving section, line position and token frequency.
6. **Segmentation robustness:** a claim survives only if its sign agrees under P70 and no-suffix analyses.

Primary measures are Jensen–Shannon context similarity, held-out codelength gain and matched-pair effect size. Five deterministic folio folds are used.

**Gate C:** at least one compositional effect must be positive in all five folds, survive both segmentations and exceed both order controls. A positive result identifies architecture only, not meaning.

### D. Open-set adjudication

Construct a recoverability-signature vector for each calibrated family from A and B: representation robustness, alignment/boundary recovery, order gain, cross-manuscript transfer and null separation. Voynich receives an architecture comparison only if Gates A and at least one of B1/B2 pass.

Use robust covariance distance and leave-one-family-out conformal ranks. A target outside the 95% support of every family is `ABSTAIN_OOD`, regardless of nearest-class probability.

## 3. Controls

External controls include GABC lyrics, monoalphabetic substitution, shuffled surfaces, first-order Markov surfaces and weak-state procedural generators. Voynich controls preserve section, line length, line position and token-frequency strata where applicable.

No target-derived threshold, model family or feature is selected after the Voynich result is inspected.

## 4. Interpretation

Possible outputs:

1. `RECOVERABLE_KNOWN_FAMILY`;
2. `RECOVERABLE_OPERATIONAL_ARCHITECTURE_UNRESOLVED`;
3. `STRUCTURE_WITHOUT_RECOVERABILITY`;
4. `ABSTAIN_OOD`;
5. `CALIBRATION_FAILURE`.

The result cannot establish plaintext, pitch, duration, recipe content, botanical identity or any glyph meaning.
