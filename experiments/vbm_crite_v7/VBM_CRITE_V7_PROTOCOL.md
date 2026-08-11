# VBM v7 — Conditional Residual Identity-Transfer Excess (CRITE)

Date: 2026-08-11
Namespace: `VBMCRITEV7`

## Scientific question

Does the VBM surface stream contain reusable latent-language/transducer information **beyond** the stable surface unigram/bigram and higher-order structure that explained most of v6.1?

v6.1 is motivation only. Its target scores and posthoc residual are not used to set v7 thresholds.

## Critical design change

The v6/v6.1 topology-preserving label-permutation intervention destroyed ordinary symbol frequencies and transition identities. v7 instead uses **exact directed-bigram-preserving Euler surrogates** independently within every retained sequence. Each surrogate preserves:

- exact sequence length;
- exact first and last symbol;
- exact symbol unigram counts (as implied by the fixed Euler trail endpoints and edge degrees);
- exact directed surface bigram multiset;
- therefore exact core/bridge symbol counts and exact counts of C→C, C→V, V→C and V→V transitions.

It randomizes Euler-trail order and therefore destroys most order ≥2 sequential organization. Exact positional C/V topology is not preserved; only the complete typed transition multiset is. This limitation is explicit and binding.

Because the v6.1 latent model itself scored only a joint surface-pair matrix, exact pair-preserving surrogates would make its score invariant. v7 therefore uses a genuinely sequential latent HMM score.

## Latent model

Candidate latent languages are Bavarian and German only.

For each outer training fold and candidate language:

1. initialize a typed emission matrix by the existing moment-factorisation fit;
2. choose the better of two deterministic moment starts by training moment loss only;
3. run exactly 8 emission-only Baum–Welch iterations with the language transition matrix and stationary distribution frozen;
4. score held-out sequences by full forward HMM log likelihood per event.

No convergence rescue, extra starts or target-specific tuning is permitted.

The candidate language with the larger training forward score after the fixed 8 iterations is frozen for that fold.

## Matched surface null

A surface-only model is selected inside each outer training fold using a deterministic inner split of training folios. Candidate models are:

- hierarchical ordinary Markov orders 1–5;
- hierarchical typed Markov orders 1–5;
- periodic/slot models with periods 2–8.

The candidate with highest inner-validation score is selected, then refit on all outer-training folios and frozen before the outer holdout is scored.

## Primary statistic

For held-out data X,

`G(X) = S_latent(X) - S_surface(X)`.

For 24 independent exact-bigram Euler surrogates X*_r generated from the same held-out sequences,

`CRITE = G(X) - median_r G(X*_r)`.

Positive CRITE means the latent transducer captures sequential information beyond both the frozen matched surface null and everything preserved by exact surface bigram counts.

Diagnostics:

- `PRED_ADV = G(X)` (raw held-out latent-vs-surface predictive advantage);
- latent and surface surrogate excesses separately;
- selected surface-null family;
- selected latent language.

No plaintext, decoded sequence or per-symbol map may be printed or inspected.

## Synthetic families

Reusable-key positives:

- `BAV_GLOBAL`
- `GER_GLOBAL`
- `BAV_GLOBAL_SWAP` (same key with sparse local same-type swaps)

Negatives/adversaries:

- `BAV_FRESH` — genuine Bavarian, fresh key per pseudo-folio
- `GER_FRESH` — genuine German, fresh key per pseudo-folio
- `MARKOV1` — stable first-order surface generator fitted to a reusable-key source
- `MARKOV2` — stable second-order surface generator
- `MARKOV3` — stable third-order surface generator
- `SLOT5` — stable periodic surface grammar with period 5

Thus a positive v7 instrument must identify reusable latent structure while rejecting genuine language with nonreusable keys and multiple stable non-language surface processes.

## Development smoke

One replicate per family, 2 outer folds, 6 Euler surrogates, reduced optimisation. Smoke is diagnostic only and may reveal implementation defects. No threshold is frozen from smoke.

After smoke, only code corrections required for correctness are permitted before formal CAL. Any conceptual/statistical redesign requires a new namespace/version.

## Formal Q0 CAL

Fresh `CAL` seeds, 3 replicates per family, 4 outer folds, 24 surrogates.

Calibration qualifies only if:

1. `min(median_CRITE reusable positives) > max(median_CRITE all negatives)`;
2. at least 8/9 reusable positive replicates have median `PRED_ADV > 0`;
3. no more than 1/18 negative replicate has median `PRED_ADV > 0` **and** median CRITE above the eventual midpoint threshold.

If criterion 1 holds, freeze

`TAU_CRITE = midpoint(min positive CRITE, max negative CRITE)`.

No target data may be scored before this freeze is committed.

## Untouched Q0 VAL

Fresh `VAL` seeds, same families and 3 replicates each.

Qualification requires:

- >=8/9 reusable positives pass `CRITE >= TAU_CRITE`;
- each reusable family >=2/3 passes;
- 0/18 negative/adversarial replicates pass;
- at least 8/9 reusable positives have `PRED_ADV > 0`.

On an irrecoverable gate failure, stop immediately and cancel compute.

## Voynich Q1 — exploratory FIT

Only if CAL and untouched VAL qualify.

Use the existing 181-folio FIT corpus. H1 and C1 are already consumed and are not used.

Use six deterministic held-folio folds. For each fold run the frozen v7 procedure with 24 exact-bigram Euler surrogates.

Exploratory FIT is called structurally positive only if:

- median CRITE >= frozen TAU_CRITE;
- CRITE > 0 in >=5/6 folds;
- median PRED_ADV > 0;
- PRED_ADV > 0 in >=5/6 folds.

Language identity is diagnostic only and cannot establish plaintext language.

## Independent-representation confirmation

A target-level confirmatory claim is **not available from VBM FIT**, because H1/C1/FIT have all been consumed by prior work. If FIT is structurally positive, the next stage must preregister an independent representation (STA/AAA or image-derived glyph stream) before inspecting its v7 statistic. No reused VBM subset may be presented as pristine confirmation.

## Stop rules

- Failed synthetic CAL or VAL => no Voynich FIT.
- Failed exploratory FIT => no independent-representation cipher claim.
- No posthoc threshold movement.
- No plaintext or mapping inspection.
- Bound paid jobs and cancel immediately on irreversible failure.
