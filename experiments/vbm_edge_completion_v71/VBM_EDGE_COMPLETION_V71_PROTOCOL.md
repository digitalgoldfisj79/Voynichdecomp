# VBM v7.1 — Held-out Surface-Pair Completion

Date: 2026-08-11
Namespace: `VBMEDGECOMP71`

## Question

Does the VBM surface bigram matrix contain a reusable **language-constrained latent factorisation** that predicts withheld surface-pair relationships better than strong non-language matrix-completion baselines?

This is the direct successor to v7, which was closed at smoke because exact-bigram-preserving Euler surrogates also preserve nearly all of a first-order homophonic cipher's likelihood.

## Core intervention

For each outer training fold:

1. aggregate the 144×144 directed surface bigram count matrix;
2. define a deterministic mask comprising 20% of eligible cells (`train count >= 3`), selected separately within the four core/bridge type blocks by a frozen hash order;
3. fit all models with those cells excluded from their fitting loss;
4. evaluate prediction only on **independent outer-held-out folio counts falling in the masked cells**.

The mask is determined from outer-training data only. Held-out counts never affect mask selection or model fitting.

## Candidate latent models

Bavarian and German language transition matrices are fixed from the pre-existing control corpora.

For each candidate language, fit the typed emission matrix `E` by masked moment factorisation of

`M = E.T @ J_language @ E`,

where `J_language = diag(pi) @ T`.

The masked cells contribute zero loss. Two deterministic starts, 700 Adam steps in formal runs. The candidate with lower masked training loss is selected.

No Baum–Welch refinement, target rescue or additional starts are allowed.

## Strong non-language baselines

All baselines see exactly the same unmasked training cells.

1. `INDEP`: outer product of unmasked row/column marginals.
2. `ALS_R8`: nonnegative weighted matrix completion, rank 8.
3. `ALS_R19`: rank 19.
4. `ALS_R32`: rank 32.

ALS uses deterministic initialization, 40 alternating ridge-regression iterations, clipping negative factors to zero, and global normalization to a joint probability matrix.

For each fold the binding non-language score is the **maximum held-out masked-cell score among all four baselines**. This holdout-side maximum is deliberately conservative: the latent model must beat every registered surface completion baseline.

## Primary score

For a model matrix M and held-out count matrix H, restricted to masked cells:

`S_mask(M,H) = sum_{masked} H_ij log(max(M_ij, eps)) / sum_{masked} H_ij`.

Primary statistic:

`EDGE_ADV = S_mask(M_latent,H) - max_b S_mask(M_baseline_b,H)`.

Positive EDGE_ADV means the language-constrained latent factorisation predicts independently observed, deliberately withheld surface-pair cells better than all registered non-language completion models.

Diagnostics:

- selected latent language;
- masked training loss;
- masked held-out edge count;
- each baseline score;
- latent masked mass and baseline masked masses.

No plaintext, state path or symbol→letter mapping may be printed or inspected.

## Synthetic families

Reusable-key positives:

- `BAV_GLOBAL`
- `GER_GLOBAL`
- `BAV_GLOBAL_SWAP`

Negatives/adversaries:

- `BAV_FRESH`
- `GER_FRESH`
- `MARKOV1`
- `MARKOV2`
- `MARKOV3`
- `SLOT5`

The non-language adversaries are generated with stable surface identities across pseudo-folios.

## Smoke

One replicate/family, two outer folds, reduced optimisation (300 masked-moment steps, 20 ALS iterations). Smoke is diagnostic only. No threshold may be taken from smoke.

Only correctness/implementation fixes may be made after smoke within this namespace. A conceptual change requires a new version.

## Formal CAL

Fresh `CAL` seeds. Three replicates/family, four outer folds.

Each replicate statistic is median EDGE_ADV across its four folds.

CAL qualifies only if:

`min(reusable positive median EDGE_ADV) > max(all negative median EDGE_ADV)`.

If separable, freeze

`TAU_EDGE = midpoint(min positive, max negative)`.

Additional calibration gate: at least 8/9 reusable positives must have median EDGE_ADV > 0.

No Voynich access before the threshold freeze is committed.

## Untouched VAL

Fresh `VAL` seeds, same family/repetition structure.

Qualification requires:

- >=8/9 reusable positive replicates pass `EDGE_ADV >= TAU_EDGE`;
- each reusable family >=2/3 passes;
- 0/18 negative/adversarial replicates pass.

Any irrecoverable failure stops the programme before Voynich.

## Exploratory Voynich FIT

Only after CAL and VAL qualification.

Use the existing 181-folio FIT set only; H1 and C1 are consumed and excluded.

Six deterministic held-folio folds. Mask construction uses each fold's training folios only. Formal v7.1 settings are frozen.

Exploratory structural pass requires:

- median EDGE_ADV >= TAU_EDGE;
- EDGE_ADV > 0 in >=5/6 folds;
- each fold has >=100 held-out masked bigram events.

Language identity remains diagnostic only.

## Confirmation status

Existing VBM data cannot provide a new pristine confirmatory holdout. If exploratory FIT is positive, a separate independent representation (STA/AAA or image-derived glyph stream) must be preregistered before its v7.1 statistic is inspected.

## Stop/compute rules

- bound HF jobs;
- cancel immediately on irreversible synthetic-gate failure;
- no threshold movement;
- no target-specific masking or model changes;
- no plaintext/mapping inspection.
