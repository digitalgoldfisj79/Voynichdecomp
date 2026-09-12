# Historical Yiddish–German L recovery calibration transfer v0.2

## RETRACTIONS / CONTROLLING LIMITS

1. v0.1 is permanently inadmissible because the ReF extraction unit was wrong.
2. v0.1b remains `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`: 7/10 at its frozen separately-standardized margin, despite strong mechanism recovery. v0.2 does not retroactively alter that result.
3. The post-v0.1b same-map paired-null diagnostic that yielded 10/10 is rejected as too optimistic because the two language solvers generally return different mappings.
4. v0.2 is a **transfer panel, not final independent confirmation**. Its Yiddish works were unseen by L v0.1b but some were consumed elsewhere in the broader R programme.
5. The six-leaf primary-script bridge remains failed. This experiment uses only the secondary PPCHY a-z representation.
6. Voynich/C7 remains sealed and must not be loaded.

Status before outcomes: `V02_EXACT_PANEL_AND_STATISTIC_FROZEN__NOT_SCORED`.

## 1. Falsifiable question

Does the recovery-based Yiddish-vs-German signal observed in v0.1b transfer to L-unseen historical families when the two returned mappings are compared in common raw audit-score units using a conservative independent-null variance?

A pass means only `L_RECOVERY_V02_TRANSFER_PASS__FRESH_CONFIRMATION_STILL_REQUIRED`. Any failed frozen gate means `L_RECOVERY_V02_TRANSFER_NOT_RESOLVED` with the failing gate named.

## 2. Frozen language models and M0 solver

Exactly inherit v0.1b except for fresh seed namespace and transfer families:

- Yiddish BUILD: `shir_1579` (`1579e-shir-preface.psd`, `1579e-shir.psd`) plus `ester_1589` (`1589e-ester-preface.psd`, `1589e-ester.psd`), balanced to exactly 10,000 words.
- German BUILD: ReF `F014` + `F015`, balanced to exactly 10,000 words.
- Yiddish representation: frozen PPCHY Romanisation, literal lowercase a-z atomizer; explicitly secondary/lossy.
- German representation: official ReF v1.0.2 CorA XML; concatenate direct-child `tok_dipl/@utf` fragments per virtual `<token>`, then literal lowercase a-z; no lemma/POS/Unicode transliteration.
- Cell: 512 fit words + 32-word separation buffer + 512 untouched audit words.
- 32 fresh planted global monoalphabetic keys per relationship family, 0% erasure.
- Search: 8 restarts × 5,000 annealing swaps + 60 greedy passes, exactly as v0.1b.
- 64 deterministic random permutation mappings per model/case for audit nulls.
- Truth recovery diagnostic: atom recovery >=0.90 AND whole-word recovery >=0.80; family mechanism pass requires >=29/32 keys.
- Fresh root, model aliases, group aliases and planted keys. No v0.1/v0.1b solver outputs may be reused.

## 3. Source-only selection freeze

Selection was frozen prospectively in `TRANSFER_PROTOCOL_PRE_CENSUS.md`, then executed in source-only workflow run `34717339316`. Artifact `10305995314` passed all source gates. The exact panel is pinned in `CENSUS_SELECTION_FROZEN.json` at commit `7533f2a92b8a8d4d7f29f84349a1caa4ec30aae3`.

No score, language-model statistic, genre or plausibility criterion entered selection.

### Yiddish transfer families — all eligible under the frozen census rule

- `1600e-magid`: 1,208 words
- `1624e-magen`: 1,904
- `1666w-messiah`: 2,821
- `1675e-ashkenaz-un-polak`: 3,052
- `1692e-vilna`: 2,207
- `1697e-purim`: 5,859
- `1705w-glikl`: 2,713
- `1750w-moses`: 2,416

### German transfer works — deterministic length matching without replacement

- `F004`: 2,950 words
- `F028`: 2,047
- `F057`: 2,875
- `F128`: 5,865
- `F166`: 1,231
- `F249`: 2,452
- `F300`: 2,018
- `F313`: 3,325

Frozen pair mapping used only for source balancing, not inferential pairing:
`magid→F166`, `magen→F300`, `vilna→F028`, `moses→F249`, `glikl→F057`, `messiah→F004`, `ashkenaz-un-polak→F313`, `purim→F128`.

Source-only gate result before scoring:

- BUILD↔TRANSFER exact 8-word overlap: zero in both languages.
- BUILD↔TRANSFER exact 5-word overlap: zero in both languages.
- cross-candidate exact 8-word overlap: zero within both languages.
- all source hashes and counts frozen before this protocol outcome run.

## 4. Blind recovery

For each of the 16 relationship families and each of 32 planted keys:

1. select a deterministic contiguous 512+32+512 span using the fresh v0.2 seed namespace;
2. encrypt fit and audit with the same hidden M0 key;
3. give the same fit ciphertext independently to anonymous model A and model B;
4. use identical search seeds and budgets for paired model runs;
5. apply each returned mapping to untouched audit ciphertext.

Solver jobs receive only the public models, opaque group ID and ciphertext. Truth language, family identity, planted key and A/B language identity remain private until all 16 solver jobs are frozen.

## 5. Frozen primary transfer statistic

For each model L in {Yiddish, German}:

- `s_L` = normalized audit bigram score under L's own returned mapping;
- `mu_L`, `sd_L` = mean and population SD of 64 random-map scores on the same audit ciphertext under L.

Define the model-baseline-centred raw contrast:

`D = (s_Y - mu_Y) - (s_G - mu_G)`

and the conservative independence-standardized contrast:

`Z_ind = D / sqrt(sd_Y^2 + sd_G^2)`.

This intentionally ignores positive same-map covariance because the two recovery solvers return distinct mappings.

Frozen target-usable call:

- Yiddish if `Z_ind >= +1.0`;
- German if `Z_ind <= -1.0`;
- otherwise ABSTAIN.

Family call = median key-level `Z_ind` across 32 keys, same ±1.0 threshold. Abstentions are errors.

No threshold, statistic or variance model may change after v0.2 outcomes exist.

## 6. Nuisance controls

### N1 — unigram frequency mapping

Use the same deterministic language-specific frequency-ranked mapping as v0.1b. On audit, compute model-specific unigram score and 64-map null mean/SD, then form **the same v0.2 contrast form**:

`D_uni = (sY_uni-muY_uni) - (sG_uni-muG_uni)`

`Z_ind_uni = D_uni / sqrt(sdY_uni^2 + sdG_uni^2)`.

Apply ±1.0 family call unchanged.

### N2 — within-word sequence destruction

Deterministically shuffle letters inside every DEVELOPMENT word before encryption, preserving word lengths, boundaries and global character counts. Run full bigram recovery and use the same `Z_ind` formula and ±1.0 call.

Neither nuisance arm can be positive evidence.

## 7. Frozen transfer gates

All seven must pass:

1. **Correct-language recovery:** every one of 16 families has >=29/32 correct-model M0 recovery passes.
2. **Wrong-language selectivity:** zero of 16 wrong-model cells reaches >=29/32.
3. **Deployable family accuracy:** >=0.80 across all 16 families, abstentions as errors.
4. **Exact label null:** accuracy effect over the exact label-permutation null (preserving 8 Yiddish / 8 German labels) is >=2.0 null SD; report effect and null SD together.
5. **Nuisance dominance:** primary makes at least two fewer family errors than N1 and at least two fewer than N2.
6. **Leave-one-family-out stability:** after dropping each family in turn, both accuracy >=0.80 and exact-label effect >=2.0 null SD remain true.
7. **Measurement completeness:** no missing/nonfinite primary case, zero/null SD or incomplete 32-key family cell.

Decision sensitivity at ±0.5 and ±1.5 may be reported after the frozen ±1.0 result but cannot change status.

## 8. Interpretation boundary

A full v0.2 pass supports transfer of the recovery-based Yiddish-vs-German discriminator across L-unseen families on the secondary normalized representation. It still does **not** issue L qualification because the Yiddish transfer families are not fresh to the entire broader programme and because the primary-script bridge failed.

Fresh external historical-Yiddish confirmation is mandatory before any target use. Wagenseil 1699 *Wieduwilt* is frozen separately as a hostile external transfer and must not be pooled into this panel.
