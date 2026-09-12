# Historical Yiddish–German L recovery competition v0.1

## RETRACTIONS / CONTROLLING LIMITS

1. Candidate 1 and Candidate 2 of the **direct classifier** family remain retired. Their nuisance-dominated results are not repaired or extended here; this is not Candidate 3.
2. The 1711 six-leaf primary-script audit failed literal/bijective inheritance. This experiment therefore operates **only on the already-qualified secondary normalized a–z representation** and cannot promote that representation to primary-script status.
3. The experiment is **DEVELOPMENT ONLY**. There is no fresh Yiddish confirmation family left in Penn at the required length, no equally qualified historical-Hebrew arm, and no target use. Voynich/C7 is sealed.

Status before outcomes: **FROZEN_PROTOCOL__NOT_SCORED**.

## 1. Falsifiable question

Can two equally budgeted historical-language models, used inside the same blind M0 substitution-recovery pipeline, distinguish held-out historical Yiddish from historical German **through fit→audit key recovery**, rather than through the shallow corpus nuisances that defeated the retired direct classifier?

The analytical distinction is explicit:

- **“Recovery-based language structure is resolved”** only if the registered primary pipeline passes the recovery, deployable-call, matched-null, nuisance and robustness gates below.
- **“Not resolved”** if any gate fails. No weaker rhetorical interpretation is allowed.

## 2. Scope and representations

- Cipher family: M0 only — one globally fixed monoalphabetic permutation of `a..z`, preserving words/order/boundaries.
- Message cell: 512 fit words + 32-word separation buffer + 512 audit words.
- Keys: 32 independently planted global permutations per relationship group, 0% erasure for L development.
- Yiddish representation: Penn Parsed Corpus historical-Yiddish Romanisation under the already-frozen a–z extractor. It is explicitly lossy/secondary.
- German representation: official ReF v1.0.2 CorA-XML diplomatic `tok_dipl/@utf`, then the same literal a–z reduction.
- Equal BUILD budget: exactly 10,000 words per language, deterministic work-family round-robin with residual redistribution.
- Same solver architecture, moves, restarts, annealing schedule and stopping rule for both languages.
- Selected solver budget is inherited from the qualified R v03 S1 configuration: 8 restarts × 5,000 annealing swaps + 60 greedy passes. It is not tuned here.

## 3. Frozen corpus groups

### BUILD models — previously consumed only

Yiddish:
- `shir_1579`: `1579e-shir-preface.psd`, `1579e-shir.psd`
- `ester_1589`: `1589e-ester-preface.psd`, `1589e-ester.psd`

German:
- `F014`
- `F015`

### DEVELOPMENT relationship groups — previously consumed only

Yiddish:
- `bovo_1507`: `1507w-bovo.psd`
- `cracow_letters_1588`: `1588e-letters-cracow.psd`
- `sam_hayyim_1590`: `1590e-sam-hayyim.psd`
- `lev_tov_1620`: `1620e-lev-tov-1-preface.psd`, `1620e-lev-tov-1.psd`
- `kine_1648`: `1648w-kine.psd`

German:
- `F016`
- `F018`
- `F034`
- `F037`
- `F148`

Relationship group, not key replicate, is the inferential unit.

## 4. Leakage and source checks before scoring

Before any outcome interpretation:

1. verify every DEVELOPMENT group has at least 1,056 normalized words;
2. verify both BUILD pools can supply exactly 10,000 words under the frozen balancing rule;
3. record hashes for all source payloads and extraction code;
4. screen BUILD↔DEVELOPMENT exact 8-word overlaps separately by language; any nonzero 8-word overlap blocks scoring until the relationship is resolved;
5. report all 5-word overlaps diagnostically;
6. verify ReF IDs from XML metadata/path, never by file order;
7. no Voynich file, transcription, score, segmentation, lexicon or statistic may be loaded.

## 5. Primary recovery competition

For each DEVELOPMENT relationship group and each of 32 planted keys:

1. take a deterministic contiguous 512+32+512 span;
2. encrypt fit and audit with the same hidden M0 key;
3. give the **same fit ciphertext** independently to the frozen Yiddish and German bigram recovery models;
4. use identical public search seeds and budgets for the paired models;
5. apply each returned mapping to the untouched audit ciphertext.

Truth/key material is never an input to either solver. It is revealed only to the independent development scorer after both model outputs are frozen.

### Truth-based mechanism diagnostic

For each model, score audit atom and whole-word recovery against truth. The correct-language model must satisfy the already-registered M0 rule (>=0.90 atoms AND >=0.80 whole words) in >=29/32 keys for **every** relationship group. No wrong-language relationship group may itself reach >=29/32 at that rule. Failure means the recovery mechanism is not language-selective enough for L.

This diagnostic is not the deployable language call because target truth would be unknown.

## 6. Target-usable audit score

For each returned mapping and language model:

- compute the normalized bigram log-likelihood of the **audit ciphertext under the returned mapping**;
- construct a matched audit null from 64 deterministic random permutation mappings on the same audit ciphertext and same language model;
- `z_L = (returned_audit_score - mean(random_map_scores)) / SD(random_map_scores)`.

If the matched-null SD is zero/nonfinite, that model/case abstains.

Primary paired margin:

`M = z_Yiddish - z_German`.

Frozen call rule:

- Yiddish if `M >= +1.0`;
- German if `M <= -1.0`;
- otherwise ABSTAIN.

The ±1.0 rule is frozen before outcomes and represents a one-null-SD separation between the two model-specific standardized audit improvements. Abstentions count as errors in development accuracy.

Family call = median key-level `M` across the 32 planted keys, with the same ±1.0 threshold.

## 7. Nuisance controls

### N1 — unigram/frequency recovery

For each language, construct the deterministic frequency-ranked monoalphabetic mapping from the fit ciphertext to that language's equal-budget BUILD unigram frequencies. Apply it to audit. Standardize the audit unigram score against 64 random mappings exactly as above. Form `M_uni = zY_uni - zG_uni` and apply the same ±1.0 call rule.

This tests whether sorted character-frequency shape alone explains the language call.

### N2 — within-word order destruction

Deterministically shuffle letters **within every plaintext word** separately for each DEVELOPMENT family before encryption, preserving word boundaries, word lengths and exact global character counts while destroying within-word sequential structure. Run the full primary bigram recovery pipeline unchanged, including the same budgets, random-map audit standardization and ±1.0 call rule.

This tests whether the primary call survives destruction of the sequence structure it purports to use.

Neither nuisance arm can be re-labelled as positive evidence.

## 8. Frozen development feasibility gates

All must pass:

1. **Correct-language recovery:** every one of the 10 relationship groups reaches >=29/32 correct-language M0 recovery passes.
2. **Wrong-language selectivity:** zero of the 10 wrong-language relationship-group cells reaches >=29/32 M0 recovery passes.
3. **Deployable family calls:** primary family-level accuracy >=0.80 across the balanced 5+5 groups, counting abstentions as errors.
4. **Matched label null:** primary accuracy effect over the exact 252 balanced-label permutation null is >=2.0 null SD. Report effect size and null SD together.
5. **Nuisance dominance:** primary makes at least **two fewer family errors** than N1 and at least two fewer than N2. If primary accuracy equals or nearly tracks either nuisance, L is unresolved regardless of nominal significance.
6. **Representation/source fragility:** no single development family may be solely responsible for crossing the primary gate; report leave-one-family-out accuracy and sign stability. If removing any one family reverses the feasibility conclusion, mark `DECISION_FRAGILE`.
7. **Measurement validity:** no missing/nonfinite primary rows, zero matched-null SD, solver exception or incomplete family cell may be treated as a pass.

Pass status, if all gates hold: `L_RECOVERY_DEVELOPMENT_FEASIBLE__CONFIRMATION_REQUIRED`.

Otherwise: `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED` with the failed gate(s) named.

Even a development pass **does not issue L qualification**. Fresh source-disjoint Yiddish confirmation, comparable historical Hebrew if a three-language claim is desired, and a separately frozen confirmation stage remain mandatory.

## 9. Decision-rule sensitivity

After applying the frozen ±1.0 rule, report family outcomes under neighbouring margins ±0.5 and ±1.5 without changing the controlling decision. This is fragility analysis only.

## 10. Audit order

Interpret only after documenting, in this order:

`circularity → leakage → confounds → matched nulls → control fairness → measurement degeneracy → representation dependence → decision-rule fragility → audit completeness`.

No target/Voynich access is authorised anywhere in v0.1.
