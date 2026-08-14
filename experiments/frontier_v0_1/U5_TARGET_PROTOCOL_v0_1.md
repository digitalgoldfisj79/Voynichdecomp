# U5-C — frozen Voynich compatibility application v0.1

Date frozen: 2026-08-14
Status: **FROZEN BEFORE ANY U5 VOYNICH SCORE IS CALCULATED**
Prerequisites: U5-A `PASS_RECOVERY_CALIBRATION`; U5-B `PASS_RECOGNITION_CALIBRATION`.

## Claim scope

U5-C does **not** test whether the Voynich Manuscript is encrypted, whether Naibbe is historical, or what any plaintext says. It tests one narrow statement:

> Does the observed Voynich token surface fall inside the operational region learned for the frozen `FRESH-VERBOSE` architecture: a fresh codebook with reusable unigram/prefix/suffix role dictionaries and combinatorial prefix+suffix compounds, as distinguished from the four preregistered matched null families?

A positive result is **mechanism-family compatibility**, not identification. A negative result excludes only this frozen architecture/representation at the calibrated scale.

The interpretation is intentionally narrower than “verbose cipher” because U5-B's largest coefficients are reusable 3-character prefix/suffix support, reusable-dictionary cost and bipartite rectangle closure. Those are structural observables that could in principle arise under another compositional production system.

## Frozen recogniser

U5-B result, before this target protocol:

- locked recall: 0.96 (96/100);
- locked precision: 1.00;
- false positives: 0/400, including 0/100 on the dependent reusable slot-morphology null;
- frozen probability threshold: `0.9997460219719421`.

The target runner must reconstruct the classifier from the exact frozen U5-B code and development sources, and **assert the selected calibration threshold equals the value above to numerical tolerance 1e-12 before reading `enriched_records.pkl` or `voynich_transcriptions_slim.json`**. Failure to reproduce the instrument aborts target opening.

No coefficient, threshold or feature may be refit after reading the target.

## Primary target representation

Primary source: canonical `enriched_records.pkl`, SHA-256 `dbf87cf5525e065da881b06a26c9d411543ff8ef3f5f8e15a9e4b557808f1174`.

Primary unit string: the record's already-frozen `token` field, in canonical record order. Empty tokens are excluded. Line/page/section/hand labels are not supplied to the recogniser.

No transliteration remapping, bench collapsing, character merging, token repair or segmentation optimisation is allowed.

## Primary block scale

U5-B positive controls begin from 4,096 plaintext characters segmented independently into one- or two-character units with equal probability. Their expected token count is `4096 / 1.5 = 2730.67`.

Therefore the primary Voynich target is partitioned **before scoring** into consecutive non-overlapping blocks of exactly **2,731 token occurrences**. With 37,465 canonical records this yields 13 complete primary blocks; the final remainder is excluded from the formal block vote but scored descriptively as a whole-corpus/remainder sensitivity.

No boundary is shifted to align with a section, hand, page or visually interesting folio.

## Target-derived destructive controls

Each primary Voynich block is accompanied by three deterministic matched controls, generated using the already-frozen U5-B transformations and seeds derived from `SHA256('u5c' || block_index || null_family)`:

1. `TYPE-RECODE`: exact token frequency spectrum and every token length preserved, each distinct type replaced by a fresh random string of the same length;
2. `GLOBAL-CHAR-RESHUFFLE`: exact atomic-character multiset and exact token-length sequence preserved, characters globally shuffled and repartitioned;
3. `TOKEN-INTERNAL-SHUFFLE`: token order, token lengths and each token's atomic-character multiset preserved, characters shuffled independently within token.

The external U5-B `DEPENDENT-SLOT3` control is not regenerated from Voynich because it requires an underlying source identity stream that is unavailable. Its locked 0/100 FPR remains an external anti-confounding qualification result.

## Formal target verdict

Let `V` be the number of the 13 primary blocks whose frozen U5-B probability is at or above the frozen threshold. Let `N_j` be the positive count for target-derived null family `j`.

- `PASS_COMPATIBLE_FROZEN_VERBOSE` iff **V >= 11/13** (≥80% on the discrete 13-block grid) **and N_j = 0/13 for every target-derived null family**.
- `FAIL_INCOMPATIBLE_FROZEN_VERBOSE` iff **V <= 2/13** (≤20%).
- otherwise `ABSTAIN_UNRESOLVED`.

The null requirement is deliberately strict: with only 13 blocks, the umbrella ≤5% FPR criterion permits zero positive null blocks.

A PASS is still only compatibility. It may motivate post-hoc adversarial morphology controls, but no such post-hoc control may alter the confirmatory U5-C verdict retroactively; it must be reported separately as a challenge to interpretation.

## Pre-specified scale sensitivities

After the formal 2,731-token result is frozen, repeat the same frozen scorer without retuning on:

- consecutive non-overlapping 2,048-token blocks;
- consecutive non-overlapping 4,096-token blocks;
- the entire canonical token corpus as one descriptive sample.

These do not replace the primary vote.

## Pre-specified transliteration sensitivity

After the primary verdict, score the five independent-family representatives already frozen in U1: `ZLZI`, `TTII`, `TTVE`, `VDRB-1`, `GCGI` from `voynich_transcriptions_slim.json`.

For each representation, concatenate normalized line-token strings in the physical page order inherited from the canonical records, partition into 2,731-token blocks, and report positive fraction and probability distribution. These are representation sensitivities only. **The best representation may not replace the primary enriched-record result.**

No family with missing/empty normalized line text is silently filled from another family.

## Reporting

Always report:

- the 13 primary probabilities and binary calls;
- all 39 target-derived destructive-null probabilities and calls;
- the formal three-way verdict;
- scale sensitivities;
- all five transliteration-family sensitivities;
- U5-B locked calibration performance beside the target result;
- the explicit statement that compatibility with reusable prefix/suffix composition is not proof of encryption, language, plaintext or provenance.
