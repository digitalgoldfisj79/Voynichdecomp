# Historical Yiddish–German L calibration transfer v0.2 — pre-census freeze

## RETRACTIONS / BOUNDARIES

1. v0.1 is permanently inadmissible because its ReF token unit was wrong.
2. v0.1b remains `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`; no v0.2 result retroactively changes it.
3. The same-map paired null that yielded 10/10 in diagnosis is rejected as too optimistic for distinct returned mappings.
4. This is a **transfer panel**, not final independent confirmation: candidate PPCHY works may have been consumed elsewhere in the broader R programme, but none may have appeared in L v0.1b BUILD or DEVELOPMENT.
5. Voynich/C7 remains sealed.

Status: `FROZEN_SELECTION_AND_STATISTIC_BEFORE_CANDIDATE_CENSUS`.

## Falsifiable question

Does the recovery-based Yiddish-vs-German signal found in v0.1b transfer to L-unseen historical families when the competing returned-map scores are compared in common raw score units with a conservative independent-null variance?

## Frozen statistic

For each language model L in {Yiddish, German} on the untouched audit ciphertext:

- `s_L` = audit bigram score under that model's own returned mapping;
- `mu_L`, `sd_L` = mean and population SD from the same 64 deterministic random-map audit null used by v0.1b.

Define

`D = (s_Y - mu_Y) - (s_G - mu_G)`

and the independence-conservative standardized contrast

`Z_ind = D / sqrt(sd_Y^2 + sd_G^2)`.

This deliberately does **not** use the positive covariance from scoring the same random mapping under both models, because the two solvers return distinct mappings.

Frozen family call, unchanged numerical cutoff from v0.1b:

- Yiddish if median key-level `Z_ind >= +1.0`;
- German if median key-level `Z_ind <= -1.0`;
- otherwise ABSTAIN.

No threshold tuning is permitted after candidate identities or outcomes are seen.

## Models and solver

Exactly inherit v0.1b:

- Yiddish BUILD model: `shir_1579` + `ester_1589`, equal total budget 10,000 words under frozen work-balancing rule.
- German BUILD model: ReF `F014` + `F015`, equal total budget 10,000 words.
- corrected ReF representation: concatenate all `tok_dipl/@utf` fragments per CorA `<token>` before literal lowercase a-z reduction.
- 512 fit words + 32-word buffer + 512 audit words.
- 32 fresh planted keys per relationship family.
- solver: 8 restarts × 5,000 annealing swaps + 60 greedy passes.
- 64 deterministic random mappings per model/case.
- same M0 truth diagnostic: atom recovery >=0.90 AND whole-word recovery >=0.80; family mechanism cell pass >=29/32.
- fresh seed namespace/root/aliases; no v0.1 or v0.1b solver outputs may be reused.

## Candidate selection — frozen before census

### Yiddish

From pinned PPCHY commit `b5864bd02a315c1d436a82553667bbf81eab6537`:

1. include only works dated <=1750;
2. merge a `-preface.psd` file with its same-work main file when both exist;
3. exclude all L v0.1b BUILD and DEVELOPMENT families: `shir_1579`, `ester_1589`, `bovo_1507`, `cracow_letters_1588`, `sam_hayyim_1590`, `lev_tov_1620`, `kine_1648`;
4. require >=1,056 pipeline words after the frozen PPCHY a-z extractor;
5. take **all** eligible remaining Yiddish families. No quality, genre, lexical, score or plausibility selection.

### German

From official ReF v1.0.2:

1. exclude v0.1b works `F014`, `F015`, `F016`, `F018`, `F034`, `F037`, `F148`;
2. require >=1,056 corrected virtual-token words;
3. after Yiddish eligibility count N is known, select exactly N German works without replacement by deterministic length matching: sort eligible Yiddish families by `(word_count, family_id)`; for each, choose the unused eligible German work minimizing absolute word-count difference, ties by lexicographic work ID;
4. no score, language-model statistic, genre or outcome may enter selection.

The census may expose only IDs, file paths, years, word counts and source hashes before this selection is frozen into a manifest.

## Leakage gates before solver execution

- exact BUILD↔TRANSFER 8-word overlap must be zero separately within each language; otherwise the affected candidate is blocked and not silently replaced;
- report 5-word overlaps diagnostically;
- report cross-candidate exact 8-word overlap and block duplicate/near-duplicate family relationships rather than treating them as independent;
- all selected source hashes and normalized counts are frozen before planted ciphertext is generated.

## Transfer gates

All must pass for status `L_RECOVERY_V02_TRANSFER_PASS__FRESH_CONFIRMATION_STILL_REQUIRED`:

1. correct-language M0 mechanism: every selected family >=29/32;
2. wrong-language selectivity: zero selected families >=29/32 under wrong model;
3. deployable family-call accuracy >=0.80 across the balanced Yiddish/German panel, abstentions errors;
4. exact balanced-label permutation effect >=2.0 null SD, effect and null SD reported together;
5. primary has at least two fewer family errors than unigram N1 and at least two fewer than within-word-shuffle N2;
6. leave-one-family-out recomputation must preserve both accuracy >=0.80 and >=2.0-null-SD call gate for every dropped family;
7. no missing/nonfinite/zero-SD primary case.

A transfer pass does not issue L. A genuinely fresh external historical-Yiddish confirmation family remains mandatory.

## External hostile transfer

Wagenseil 1699 *Wieduwilt* is frozen separately. Its result cannot be pooled into the primary panel because its Latin transliteration convention is not representation-equivalent to PPCHY.