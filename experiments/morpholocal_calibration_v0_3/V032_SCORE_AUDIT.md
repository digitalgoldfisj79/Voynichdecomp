# Morpholocal calibration v0.3.2 score audit

Date: 2026-07-15

## Scope and provenance

This audit analyses the completed selector-parity recalibration at commit
`23d54f9475c16ad7ce8a187130220c81885b645c`.

The merged dataset contains 480 solver trials:

- 288 synthetic positives;
- 192 non-cipher controls;
- beam, heuristic and parallel-tempering best-state optimisation;
- 96 positives and 64 controls per solver.

Flattened trial table:

- temporary artifact: `https://n.uguu.se/qZmmfEJw.gz`
- decompressed CSV SHA-256: `1b3a8342e25e0e0d4dc21ec338a7f4fbafed1dd140e63fc675f8ce261b08bc9d`

Artifact manifest:

- temporary artifact: `https://h.uguu.se/dzdDkgpR.gz`
- decompressed JSON SHA-256: `b7efb089d208f1e85563a4e0595ff4a6918fa786d34fd39e7034e9c0ac558e86`

Core threshold results:

- temporary artifact: `https://h.uguu.se/pgjVJiOF.gz`
- decompressed JSON SHA-256: `bc7f145e5de521a7cad6b350571437e71cede414ee52c0a2c34fdac6745edc0c`

Restricted-feature design:

- temporary artifact: `https://n.uguu.se/LXqnEjsc.gz`
- decompressed JSON SHA-256: `633c5928700dda9a2573c5c7082a05c7586188cb73d641698c2f3c531e558d9d`

## Findings

### Held-out predictive gain does not separate the classes

The held-out cipher-minus-production gain has AUC 0.534469. Median predictive
gain is 0.02970 bits per test token for positives and 0.10419 for controls.
Controls therefore have the larger median apparent cipher advantage.

A grouped threshold fitted on training folds yields:

- sensitivity 35.76%;
- specificity 67.71%;
- balanced accuracy 51.74%.

Constraining training false positives to at most 5% yields:

- sensitivity 17.71%;
- specificity 95.83%;
- balanced accuracy 56.77%.

No solver or solver ensemble materially rescues this score. Per-solver gain
AUCs are 0.5583 for beam, 0.5552 for heuristic and 0.4954 for parallel
tempering. The best simple solver ensemble reaches AUC 0.5732.

### Ordinary grouped validation is misleading

A regularised logistic model using only observable, normalized evidence can
interpolate among the represented generator families. In ordinary grouped
cross-validation, with a 5% training false-positive constraint, it reaches
approximately 69.8% sensitivity and 94.3% specificity.

This performance does not generalise to a new control mechanism.

### Generator-disjoint validation fails on permuted ciphers

The decisive test simultaneously withholds one positive selection policy and
one complete control family. When `permuted_cipher` is absent from training,
the restricted evidence model falsely accepts about 42--43 of 48 permuted
controls, depending on the held-out positive policy. Specificity falls to
roughly 10--12.5% despite the 5% training false-positive constraint.

An extended model using more score components also fails: it falsely accepts
46 of 48 unseen permuted controls, giving specificity 4.17%.

Fixture-level aggregation across solvers does not resolve the failure.

### Interpretation

The current evidence vector detects recoverable partitions and generator
fingerprints. It does not establish that the recovered latent symbols occur in
message-bearing order. A permuted-cipher control preserves cipher-like surface
and partition structure while destroying the latent sequence, and the gate
cannot distinguish it from a positive cipher.

## Verdict

**NO THRESHOLD RESCUE. REDESIGN REQUIRED.**

The existing H/I conjunction, held-out gain, learned composite scores and
solver ensembles must not be applied to the Voynich Manuscript. The next test
must condition on the fitted partition and directly test whether held-out
latent order has significantly better external-sequence likelihood than
constrained within-line randomizations preserving line length and latent-unit
composition.
