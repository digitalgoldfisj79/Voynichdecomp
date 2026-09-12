# Historical Yiddish–German L recovery competition v0.1b

## RETRACTIONS / CONTROLLING LIMITS

1. **v0.1 is permanently inadmissible.** GitHub Actions run `34714545374` used the correct seven ReF XML payloads but parsed each `tok_dipl` fragment as a separate word. No v0.1 solver output, aggregate, call, score, threshold crossing, or apparent result may be used as evidence here.
2. **Retraction:** commit `0f9e30ee32185cc65591b2da4867fa3bb8b23574` did not contain a committed ReF XML tarball. It configured an Actions workflow intended to package/upload one as an artifact.
3. Candidate 1 and Candidate 2 of the retired direct-classifier family remain retired. v0.1b is a recovery experiment, not Candidate 3.
4. The YidTakNL 1711 six-leaf primary-script bridge failed literal/bijective inheritance. v0.1b therefore remains development-only on the already-used secondary normalized `a-z` representation. It cannot promote that representation to primary-script status.
5. Voynich/C7 remains sealed. No Voynich file, transcription, statistic, segmentation, lexicon, or score may be loaded.

Status before outcomes: **FROZEN_REPAIR_PROTOCOL__NOT_SCORED**.

## 1. Parent protocol and permitted repair

The scientific design is inherited without outcome-dependent modification from `experiments/yiddish_l_recovery_v01/PROTOCOL.md` (blob `a7f350c6ef55f1c617dbf9cdfca173316f21b5bf`).

The **only permitted scientific repair** is the German ReF extraction unit. Corpus identities, BUILD/DEVELOPMENT assignment, Yiddish extraction, model class, solver architecture, solver budget, fit/buffer/audit sizes, number of keys, matched-null size, nuisance arms, thresholds, family-level inferential unit, leakage gates, robustness gates, and interpretation boundaries are unchanged.

No v0.1 solver artifact is an input to v0.1b. v0.1b creates a fresh public preflight, fresh private truth/root, fresh encrypted cases, reruns all ten blinded solver jobs, and aggregates only those fresh outputs.

## 2. Correct ReF diplomatic extraction rule

For each frozen ReF work ID, select the XML whose SHA-256 equals the preregistered Candidate-2 source hash. Then:

1. iterate each CorA `<token>` element in document order;
2. within that token, concatenate in order the `utf` values of all **direct child** `<tok_dipl>` elements;
3. lowercase the concatenated string;
4. retain only literal ASCII characters `a` through `z`;
5. append the resulting word iff nonempty.

The unit is therefore the CorA virtual `<token>`, **not** an individual `tok_dipl` fragment.

No Unicode transliteration, `tok_anno`, lemma, POS, modernisation, dehyphenation heuristic, or linguistic repair is allowed.

## 3. Hard source-equivalence gate

The preflight MUST abort unless both SHA-256 and extracted word count equal the frozen Candidate-2 values for every work:

| ReF ID | SHA-256 | required words |
|---|---|---:|
| F014 | `059211745db8c12b96d9ac4538cd94201c758f47cbc756b5fca70eeb59872f76` | 19097 |
| F015 | `29b07d566cc8f3ceedbac16a5a0f98a9a4d8059b748c53d336d97aaffcd250f9` | 18162 |
| F016 | `b898a98189fe8b3d20867dd347f71639643ee609513b76d5d85f19048bf80284` | 8124 |
| F018 | `c694913a336563a5526b5447279359ac822da86e9ecf9ea50d1b30857de8368b` | 6270 |
| F034 | `9d4e2da5fe4ac5f0438fd7599234749f4b3fe02333f60f19fed1e27215829135` | 5663 |
| F037 | `8f7e855a96c3a6fcf03c4d91102197ddbd9b93fc79ab94393854f35385825f7d` | 15327 |
| F148 | `4ca05d5794994980356dd3f9c932f8403a42d964745950dd1966d121d36c2802` | 20282 |

Reference freeze: `experiments/yiddish_l_development_20260912/candidate2_pre_scoring_gate.json`, whose German representation explicitly specifies diplomatic fragments concatenated per virtual token.

Failure of any one hash or count makes the run **inadmissible** and blocks solver jobs.

## 4. Inherited fixed design

- M0 family: one global monoalphabetic permutation of `a..z`, word/order/boundary preserving.
- BUILD budget: exactly 10,000 words per language.
- FIT: 512 words.
- separation buffer: 32 words.
- AUDIT: 512 words.
- 32 independently planted keys per relationship group.
- 5 held-out Yiddish and 5 held-out German relationship groups; relationship group is the inferential unit.
- solver: inherited R-v03 S1, 8 restarts × 5,000 annealing swaps + 60 greedy passes.
- matched audit null: 64 deterministic random mappings per model/case.
- deployable primary call: `M = z_Yiddish - z_German`; Yiddish if `M >= +1.0`, German if `M <= -1.0`, otherwise abstain.
- family call: median key-level M over 32 keys, same threshold.
- recovery diagnostic: atom recovery >=0.90 AND word recovery >=0.80; required in >=29/32 keys for every correct-language relationship group; no wrong-language cell may reach 29/32.
- N1: frequency/unigram mapping control.
- N2: within-word letter-shuffle control.
- BUILD↔DEVELOPMENT exact 8-word overlap must be zero separately within each language; 5-word overlap is diagnostic.

The deterministic public seed namespace remains `yiddish_german_l_recovery_v01_20260912`, so v0.1b does not silently change span selection, solver search seeds, nuisance shuffles, or random-map nulls merely because the repair is renamed. The private HMAC root is freshly generated at preflight, exactly as a clean rerun of v0.1 would do.

## 5. Frozen feasibility gates

All inherited gates must pass:

1. all 10 correct-language relationship groups have >=29/32 M0 recovery passes;
2. zero wrong-language relationship-group cells have >=29/32 passes;
3. primary family accuracy >=0.80, abstentions counted as errors;
4. primary accuracy effect over the exact balanced-label permutation null >=2.0 null SD;
5. primary makes at least two fewer family errors than N1 and at least two fewer than N2;
6. leave-one-family-out conclusion stable under the inherited rule;
7. no missing/nonfinite primary row, zero matched-null SD, solver exception, or incomplete family cell is a pass.

Decision sensitivity at ±0.5 and ±1.5 is diagnostic only and cannot alter the frozen ±1.0 decision.

If every gate passes: `L_RECOVERY_DEVELOPMENT_FEASIBLE__CONFIRMATION_REQUIRED`.

Otherwise: `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`, with failed gates named.

Even a pass does **not** issue L qualification. It only licenses a separately frozen, fresh source-disjoint confirmation stage. It never licenses C7/Voynich scoring by itself.

## 6. Audit order

Interpret only after documenting:

`source equivalence → circularity → leakage → confounds → matched nulls → control fairness → measurement degeneracy → representation dependence → decision-rule fragility → audit completeness`.
