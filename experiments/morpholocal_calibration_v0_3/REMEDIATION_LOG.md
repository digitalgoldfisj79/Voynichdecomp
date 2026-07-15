# Morpholocal calibration v0.3.1 remediation log

**Status:** development remediation in progress  
**Parent review brief:** `41cf32137fd19055c80bda090cb2df966a680726`  
**Working branch:** `experiment/morpholocal-calibration-v0.3.1-remediation-20260715`  
**No Voynich manuscript data are authorised or used.**

## Trigger

GPT-5.6 Sol Pro returned `REDESIGN_BEFORE_MORE_COMPUTE`, principally because the existing development acceptance layer produced approximately 50% shared false positives. The review was treated as a design audit rather than accepted mechanically.

The immediate remediation objective is to determine whether the shared false positives arose from implementation contamination, production-model weakness, accounting asymmetry, or genuine overlap between the declared positive and control classes.

## Implementation defects found during remediation

### 1. Sequence corruption in length subsetting

`tournament_runner.py::subset_events` sorted selected events using `token_index`. The development log itself records that `token_index` is a vocabulary index rather than an event-position index. Multiple `MID` events could therefore be reordered within a line.

Remediation: `remediation_runtime.py::safe_subset_events` selects complete lines and returns them in exact source-list order.

### 2. Unsafe KT policy cache

`tournament_kt.py::prepare` cached by `id(events)` without retaining the source object. Python may recycle object ids after garbage collection, allowing a later same-process trial to receive arrays prepared for an earlier trial.

Remediation: bounded LRU retaining the exact list object and checking identity before reuse.

### 3. Unsafe production-null cache

`production_null_registry.py::rich_production_predictive_nll` cached by `(id(train), len(train))` without retaining the training list.

Remediation: bounded identity-safe LRU retaining the exact list object. Production selection remains training-only and test probabilities remain frozen from training counts.

### 4. Unsafe label-event cache

`tournament_fast.py::cached_label_events` cached by `(id(train), scheme, label)` without retaining the training list. This remained active under the KT wrapper because `fit_candidate` calls `base.label_events`.

Remediation: bounded identity-safe label cache in `remediation_runtime.py`.

### 5. Unsafe fast-policy cache on the neural path

`tournament_fast.py::prepare` cached policy arrays by `id(events)` without retaining the event list. The classical KT path replaces this likelihood implementation, but the neural path uses it directly.

Remediation: `neural_remediation_runtime.py` supplies identity-safe policy and label caches, exact-order subsetting, and the charged production-null registry.

## Regression tests

`remediation_tests.py` covers:

1. exact within-line order preservation despite deliberately scrambled vocabulary indices;
2. KT prepare-cache isolation for distinct same-length lists;
3. label-event cache isolation for distinct same-length lists;
4. production-null cache isolation under matched contexts;
5. bounded cache sizes.

Clean-container passing jobs:

- `6a576b2885d9643ce16d411b` — initial four-test pass;
- `6a576b8885d9643ce16d413b` — audit-instrumented pass;
- `6a576d42b1669a49bf074603` — five-test pass including label cache.

End-to-end one-control smoke:

- `6a576d5785d9643ce16d41b1` — passed; control rejected; complete trial audit emitted.

## Superseded/cancelled audits

The following 64-control jobs were cancelled immediately after the remaining unsafe label cache was identified. Their outputs are invalid and must not be interpreted:

- PT: `6a576c6d85d9643ce16d417c`;
- heuristic: `6a576c7b85d9643ce16d417e`;
- beam: `6a576c89b1669a49bf0745c5`.

## Active corrected classical audits

All use seed `3030303`, 64 controls, identical length assignment, 24 `cpu-xl` workers, the charged four-null production registry, exact-order subsetting and identity-safe caches. They cloned commit `2939840f576982c026db27cc158b5496ed6023d2` before later documentation/neural-launcher commits.

- PT/best-state optimiser: `6a576d86b1669a49bf074619`;
- heuristic: `6a576d96b1669a49bf07461b`;
- beam: `6a576da785d9643ce16d41c5`.

Artifacts are uploaded to private dataset `Digitalgoldfish79/voynich-morpholocal-v031-remediation` under commit-specific paths.

## Audit fields added per trial

- selected production null;
- production train and held-out costs;
- production model-index charge;
- cipher training selection score;
- inferred cipher held-out cost;
- held-out cipher-minus-production difference and per-token value;
- legacy and scientific solver labels;
- selected scheme, null count, size profile, external profile, policy and selector;
- original acceptance decision;
- strict counterfactual requiring actual held-out cipher advantage.

## Neural checkpoint status

The original trained checkpoint had SHA-256:

`578dcdd948d80d172cfa62742b70a3e9fb14a8afe80669fb7bdd07b28fbc3461`

It was uploaded only to temporary Uguu URL `https://h.uguu.se/TONOijtm.gz`, which now returns HTTP 404. Completed Hugging Face job files are not preserved as retrievable artifacts. The training code did not explicitly seed PyTorch/DataLoader RNGs, so a fresh training run cannot be represented as the exact same checkpoint.

Accordingly, no neural rerun will be described as a controlled checkpoint rerun unless the exact file is recovered. `neural_remediation_runtime.py` is prepared for such a run and may also be used later for a clearly labelled newly trained model.

## Decision interlock

No full positive-grid rerun is authorised until the corrected 64-control audits are complete and the trial-level false-positive overlap is analysed. No formal seeds have been opened.
