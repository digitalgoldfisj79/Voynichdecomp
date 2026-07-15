# GPT-5.6 Sol Pro independent review brief

## Purpose

Independently assess whether the current v0.3 path is methodologically correct before further development or formal calibration. Do not optimise for agreement with the existing team. Identify invalid assumptions, missing controls, leakage risks, accounting asymmetries, and better alternatives.

## Programme status

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/morpholocal-calibration-v0.3-20260715`
- v0.2 closure commit: `e203d5f1a69f618297c630fcb30f209accc14343`
- v0.3 protocol freeze: `38d130b1242e1542dde759e867fe88f086ae3367`
- Production-null registry commit: `779c662b2c6278865ac362d32a1402075b7bd0d5`
- No Voynich manuscript data have been used in v0.3 development.

## Scientific question

Can a literature-aligned decoder tournament reliably recover a bounded mixed-unit homophonic nomenclator class while rejecting matched production-only controls?

V0.3 intentionally retains the v0.2 synthetic class and replaces the weak inference stack with:

1. specialised policy-aware heuristic search;
2. constrained beam search;
3. parallel-tempering Bayesian/Metropolis search;
4. a permutation-equivariant synthetic-trained graph decoder;
5. explicit mixed-unit historical lattice and higher-order n-gram reranking.

## Benchmark

- 24 surface cells.
- 12 payload units, optionally extended by two null units.
- Global or Currier-style partitioned keys.
- Balanced or unequal homophone classes.
- Word-heavy, balanced, or letter-heavy external profiles.
- Selection policies: iid uniform, cyclic, frequency-weighted, sticky line-reset.
- Surface selectors: none or adjacent-length.
- Length strata: 2,000, 8,000, and 36,000 events.
- External historical corpus: 52,004 words.
- Development set: 96 positives and 64 controls, seed `3030303`.
- Controls: context-iid, cell-Markov, copy-mutate, and permuted-cipher.

## Development results

### Heuristic

- Positive recovery: 28/96.
- Median mapping accuracy: 0.7604.
- Median latent-unit error: 0.2131.
- False positives: 32/64.
- Cyclic recovery: 2/24.

### Bayesian

- Positive recovery: 57/96.
- Median mapping accuracy: 1.0000.
- Median latent-unit error: 0.0000.
- Policy recovery: 0.8542.
- Structure recovery: 0.7917.
- False positives: 35/64.
- Policy strata: iid 20/24, frequency-weighted 18/24, sticky 17/24, cyclic 2/24.
- Length strata: short 11/32, medium 24/32, long 22/32.

### Beam

- Positive recovery: 7/96.
- Median mapping accuracy: 0.4688.
- Median latent-unit error: 0.5309.
- False positives: 35/64.

### Neural

- Training: A100, 6,000 independent synthetic trials, random keys and per-example cell permutations, 45 epochs.
- Validation constrained mapping accuracy: 0.8758.
- Positive recovery: 53/96.
- Median mapping accuracy: 0.9167.
- Median latent-unit error: 0.0076.
- Policy recovery: 0.7813.
- Structure recovery: 0.7917.
- False positives: 32/64.
- Policy strata: iid 20/24, frequency-weighted 15/24, sticky 17/24, cyclic 1/24.
- Length strata: short 8/32, medium 21/32, long 24/32.

## Overlap finding

False positives are strongly shared, not independent:

- Bayesian and neural agree on 50/96 positive selections.
- Bayesian and neural also agree on all 32 neural false positives.
- All four solvers agree on 23/64 controls.

This indicates a common decision/accounting defect rather than merely independent search errors.

## Current diagnosis

The key-search engines, especially Bayesian and neural, materially improve recovery over v0.2. The present bottleneck is the cipher-versus-production acceptance layer.

The cipher side has:

- explicit latent transition likelihood;
- explicit homophone-selection policy likelihood;
- charged policy identity;
- mapping and structure costs;
- held-out scoring.

The previous production side retained a weaker v0.2 surface model. A new development-only production-null registry was therefore added with four charged held-out alternatives:

1. context-conditioned iid;
2. cell Markov;
3. context-conditioned cell Markov;
4. repeat/copy production.

The strongest production model is selected with an explicit model-index charge. A smoke test is being run before a full Bayesian development rerun.

## Historical lattice result

On a separate explicit historical mixed-unit fixture:

- 128 candidate mappings were generated.
- Best mapping accuracy among candidates: 0.7590.
- Higher-order 3/5/6-gram reranking selected a mapping with accuracy 0.7229.
- Latent-unit error: 0.0696.
- Character TER: 0.1113.
- The best bigram mapping had accuracy 0.7590 but worse latent error and TER.

This suggests higher-order reranking improves output reconstruction even when exact key-cell accuracy is not maximal.

## Known unresolved weaknesses

1. Cyclic policy recovery remains near zero.
2. Short-text recovery remains inadequate.
3. The development runners still load the v0.2 compatibility implementation dynamically; formal execution requires a static patch-free effective source.
4. Output-level thresholds are not yet frozen.
5. No formal seeds have been opened and no formal calibration has run.
6. The new production-null registry has not yet completed full development evaluation.

## Questions for independent review

Provide a hostile, technically specific answer to each question.

1. Is the diagnosis correct that v0.3 has largely improved inference but exposed a common model-comparison defect?
2. Is replacing the old production baseline with a charged held-out production-null tournament the correct immediate next step?
3. Are the four proposed production nulls sufficient and fair? Specify any missing nulls required before formal calibration.
4. Does selecting the best production null with a `log2(K)` model-index charge constitute a valid comparison, or should production models be mixed/marginalised differently?
5. Is the cipher side still receiving uncharged flexibility relative to the production side? Inspect likely costs: mapping, partition, unit-profile, external-profile, null count, selection policy, selector, language model, search over candidate structures, and neural checkpoint identity.
6. Should the acceptance decision be based primarily on held-out predictive performance, universal/KT coding, Bayes factors, cross-validation, or a conjunction? Give an exact recommended rule.
7. How should the search stage be separated from the hypothesis-test stage to avoid selection bias from choosing the best mapping on the same data used for model comparison?
8. Is a nested split required: search/train, model-selection validation, and final held-out test? Specify proportions or document-level splitting.
9. How should synthetic controls be expanded so that the decoder cannot exploit generator artefacts?
10. How should cyclic-policy failure be handled? Is the current cyclic generator too brittle/unhistorical, or does the decoder require a distinct deterministic-state inference method?
11. Are 2,000-event short trials historically meaningful, and what minimum length curve should be required?
12. Does the separate higher-order historical lattice track genuinely address the literature-review requirement, or is it insufficiently integrated with the compatibility benchmark?
13. Should the neural decoder remain in the formal ensemble, given that it was trained on the same declared generator family, even with disjoint keys and permutations?
14. What precise development result would justify freezing v0.3 for formal calibration?
15. Give a ranked next-step plan, identifying any step that should be stopped or redesigned.

## Required verdict format

Return:

1. `PATH_VERDICT`: one of `CONTINUE_AS_PLANNED`, `CONTINUE_WITH_REQUIRED_CHANGES`, `REDESIGN_BEFORE_MORE_COMPUTE`, or `STOP_V0_3`.
2. Five most important findings, ranked.
3. Exact required changes before the next full development run.
4. Exact formal-freeze conditions.
5. Whether the bounded historical cipher class remains meaningfully open after the current results.

Do not infer anything about the Voynich manuscript itself. This is exclusively a synthetic calibration and methodology review.
