# Terminal Cipher Programme v1 — frozen protocol

Date frozen: 2026-08-09
Branch: `experiment/cipher-coverage-closeout-v1-20260809`
Parent audit: `COVERAGE_AUDIT_GATE1.md`, commit `4420a84b2385189edf5a10bb8dc7fa91dd165e85`
Status: **FROZEN BEFORE QUALIFICATION OR VOYNICH SCORING**

## 1. Objective

Close only the three bounded residues admitted by Gate 1:

- **M0** — simple global monoalphabetic substitution/re-labelling evidence bridge;
- **TQ** — medieval within-word/local transposition, composed where necessary with one global monoalphabetic substitution;
- **NQ** — deterministic one-null-per-word insertion schedules, composed with one global monoalphabetic substitution.

No other cipher family may be added after execution starts. A failed arm stops; optimizer, rule inventory, language panel, representation, thresholds and nuisance ranges may not be changed in response to Voynich performance.

This programme separates **recovery**, **positive compatibility**, and **recognition specificity**. A mechanism may be rejected as incompatible if a converged best fit falls below an absolute positive-control floor. A positive/candidate interpretation additionally requires blind specificity against structured non-message controls. Beating shuffled controls alone is never sufficient.

## 2. Historical anchors

### TQ

The finite local-transposition inventory is anchored in the medieval Arabic cryptological tradition associated with Ibn al-Durayhim (1312–1361), preserved/discussed by al-Qalqashandi (`Subh al-a'sha`, completed 1412), and treated by C. E. Bosworth, "The Section on Codes and Their Decipherment in Qalqashandi's Subh al-A'sha", *Journal of Semitic Studies* 8.1 (1963), 17–33, DOI `10.1093/jss/8.1.17`.

Treccani's *Storia della Scienza* synthesis explicitly records reversal and bounded within-word permutations, including the outside-in examples `1234567 -> 1726354` and `1234567 -> 7162534`, and separately records composite transposition + substitution. Fixed-width vertical/columnar transposition is excluded because it was already covered by the repository.

### NQ

The same Treccani synthesis states that Ibn al-Durayhim's repertoire includes adding extraneous letters to the original text, including inserting an extraneous letter at a position inside every word and sequences of changing added letters across successive words. The source does not supply one uniquely obligatory insertion rank. Therefore v1 tests only a finite global-rank formalisation; position-changing or plaintext-dependent insertion is classified non-identifiable and excluded.

Historical source URL (for provenance only):
`https://www.treccani.it/enciclopedia/la-civilta-islamica-condizioni-materiali-e-intellettuali-criptologia-e-criptoanalisi_%28Storia-della-Scienza%29/`

## 3. Frozen Voynich representation

Exactly one primary target representation is permitted.

Source:
`https://www.voynich.nu/data/RF1b-er.txt`

Source-only preflight before protocol freeze:

- bytes: `343891`;
- SHA-256: `eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2`;
- parsed words before core-vocabulary exclusion: `38496`;
- alphabetic positions: `194619`.

Parsing:

- IVTFF metadata is not text;
- `.` is a certain word boundary;
- `<->` and `<~>` imply certain word boundaries;
- `,` is an uncertain space and does **not** split a word (long-word policy);
- bracketed alternative/uncertain readings cause the affected word to be excluded rather than selecting an alternative;
- locus/page identity is retained only for deterministic folio splitting.

Core surface alphabet is defined prospectively as every lower-case ASCII letter occurring at least 100 times in the source-only census. It is exactly:

`a c d e f g h i k l m n o p q r s t y`

19 symbols. The six lower-frequency letters (`b,j,u,v,x,z`) total 115/194619 alphabetic positions. Any word containing a non-core or uncertain character is excluded whole. The binding retained-character coverage requirement is >=0.995.

No STA-family, full-STA, connected-aaa, ZL, IT, GC, Takahashi or alternative EVA arm may be introduced to rescue a failed result. Independent transliteration replication is permitted only after a fully confirmed candidate and cannot rescue primary failure.

## 4. Frozen plaintext alphabet and language panel

All plaintext controls and language models use the 19-letter historical-normalisation alphabet:

`abcdefghilmnopqrstu`

Normalisation:

- Unicode NFKD / transliteration to Latin where required;
- lower case;
- `j -> i`;
- `v,w -> u`;
- `y -> i`;
- `x,z -> s`;
- discard all remaining characters outside the 19-letter alphabet;
- preserve word boundaries.

Frozen candidate-language panel, inherited from the M19 programme:

1. Latin — UD Latin ITTB;
2. Italian — UD Italian ISDT;
3. German — UD German GSD;
4. French — UD French GSD;
5. Ancient Greek — UD Ancient Greek Perseus;
6. Hebrew — UD Hebrew HTB;
7. Arabic — UD Arabic PADT;
8. Spanish/Castilian — UD Spanish AnCora.

The corpus URLs are exactly those pinned in `bnf_free_switch_m19_v0_7/run_m19.py`. Sentence-index residues `{0,1,3,4,6,8}` train language models; residues `{2,5,7,9}` are control-only and never train the model.

**Historical positive-control anchor:** UD Latin ITTB is a medieval scholastic Latin treebank based principally on Thomas Aquinas. Latin ITTB controls are binding for every exact structural rule. Other languages test transfer/generalisation and prevent a Latin-only instrument from being promoted as universal.

## 5. Frozen structural inventory

All transformations operate independently inside pre-existing words. Word order and word boundaries are never permuted or inferred.

### M0

`ID`: identity word order. A fresh global bijection maps the 19 plaintext symbols to 19 opaque cipher symbols.

### TQ — exact five-rule inventory

For a plaintext word with positions `1..n`:

- `TQ_REV`: reverse every word;
- `TQ_LAST_FIRST`: move the last letter to the first position;
- `TQ_SWAP_ENDS`: swap the first and last letters, leaving the interior fixed;
- `TQ_OUTSIDE_L`: output `1,n,2,n-1,3,n-2,...` (the generalisation of `1234567 -> 1726354`);
- `TQ_OUTSIDE_R`: output `n,1,n-1,2,n-2,3,...` (the generalisation of `1234567 -> 7162534`).

Words too short for a rule are transformed by the same mathematical permutation, which may reduce to identity.

For target use, every TQ rule is followed by one global monoalphabetic bijection. This is necessary because Voynich surface glyph names have no independently known identity correspondence to Latin letters. It is also historically licensed by the attested composite transposition+substitution direction. No per-word, per-line, per-section or per-folio substitution key is allowed.

### NQ — finite global-rank inventory

Each plaintext word receives exactly one extraneous ordinary alphabet symbol before the global monoalphabetic bijection. Its identity is sampled independently from the 19-symbol plaintext alphabet and carries no information. The insertion position is determined by one document-global rule.

Ten rules are frozen:

- offsets `L0,L1,L2,L3`: insert at the 1st, 2nd, 3rd or 4th available interior slot counting from the left;
- offsets `R0,R1,R2,R3`: analogous rank from the right;
- `MID_FLOOR` and `MID_CEIL`.

For short words, ranks are clipped deterministically to the nearest valid insertion slot. The inverse candidate deletes exactly one symbol per word at the corresponding position.

This inventory is a bounded formalisation of the source-described "one extraneous letter inside every word" operation. It does not claim that Ibn al-Durayhim prescribed these ten ranks. Free per-word insertion positions, semantic positions, content-conditioned positions, multiple arbitrary nulls and omission/lossy deletion are outside v1 and remain NON-IDENTIFIABLE.

## 6. Statistical model and solver

### Language model

For each language, train-only word-internal character bigram probabilities plus word-start, word-end and unigram probabilities are add-0.25 smoothed. Scores are total log likelihood divided by retained plaintext characters. No neural model is used.

### Global substitution solver

For each candidate structural inverse and language:

1. transform/delete the ciphertext words according to the frozen candidate;
2. build sufficient statistics: 19x19 adjacent-symbol counts, start counts, end counts and unigram counts;
3. initialise a bijective cipher->plaintext map by frequency rank;
4. optimise only by pair swaps using simulated annealing with exact score deltas;
5. use two independent ensembles A/B;
6. each ensemble runs batches of four restarts, 60,000 legal swap proposals per restart;
7. after 4, 8, 12 and 16 restarts per ensemble, stop if best A/B held-fit objective differs by <=1e-7 nats/character and occurrence-weighted map agreement is >=0.95;
8. maximum 16 restarts per ensemble.

No optimizer amendment is permitted after qualification starts. A target fit that does not converge cannot be used to reject the mechanism; it is `UNRESOLVED_SEARCH`.

### Recognition versus recovery

- **Recovery** = normalized plaintext-character accuracy under a known synthetic truth after structural inversion and map solving.
- **Structural recognition** = selecting the correct frozen structural rule without oracle rule information.
- **Language recognition** = selecting the correct language from the eight-language panel.
- **Positive compatibility** = a fixed-key held-out score reaching the absolute positive-control floor.

These are reported separately.

## 7. Qualification stages

Voynich text may be downloaded/censused for source integrity and representation coverage, but **no Voynich language/cipher score may be computed until Q1 and Q2 pass for the relevant family**.

### Q1 — exact-rule recovery qualification

Fresh controls use control-only corpus residues and fresh global mono keys.

For every exact rule in M0/TQ/NQ:

- two Latin-ITTB controls;
- one additional control whose language is assigned deterministically by SHA-256 over the rule name across the other seven languages;
- 768 plaintext letters minimum per fit half and 768 per held-out half;
- independent A/B ensembles.

Binding rule-level gates:

- median held-out plaintext recovery >=0.95;
- minimum >=0.85;
- A/B occurrence-weighted map agreement >=0.90 for every control;
- convergence reached for every control.

Every rule in a family must pass. A failing rule closes that family as a blind finite family; it may not be deleted post hoc.

### Q2 — absolute positive-control calibration

For each of the eight languages and each family M0/TQ/NQ, create three fresh controls under deterministically selected rules (identity for M0; balanced rule rotation for TQ/NQ), disjoint from Q1.

Using the true structural rule but not the key:

- recover the global map from the fit half;
- freeze it;
- score the held-out half;
- record correct-language score, runner-up scores, recovery and A/B agreement.

Binding Q2 gates per family:

- all 24 controls converge;
- median plaintext recovery >=0.95;
- minimum recovery >=0.80;
- true language ranks first in at least 22/24 controls;
- every language has at least 2/3 correct ranks;
- median true-v-runner-up margin >=0.05 nats/character.

After Q2 passes, freeze for each `family x language` the **5th percentile** of its three held-out positive-control scores using linear interpolation. This is the absolute positive-compatibility floor. No target score may change these floors.

Q1+Q2 passing is sufficient to permit a **negative compatibility test** on Voynich. It is not sufficient for a positive cipher claim.

### Q3 — blind recognition/specificity gate

Q3 is mandatory only for a positive/candidate interpretation.

Locked positives: 24 new controls (8 languages x 3 families), with unseen rules/keys and no oracle structural rule. Search the complete 16-rule inventory and all eight languages.

Locked structured negatives: 64 controls, equally divided among iid-unigram, order-2 Markov, motif-repeat/mutate and copy-mutate generators, with opaque mono relabelling and matched length/word-length support.

A blind positive is accepted only if its selected candidate reaches that candidate's frozen Q2 absolute floor.

Q3 gates:

- family accuracy >=20/24;
- exact-rule accuracy >=18/24;
- language accuracy >=20/24;
- median plaintext recovery >=0.90;
- structured-negative false-positive rate <=2/64;
- no negative generator family has >1 false positive.

If Q3 fails, Voynich may still yield a formal **negative** when its best converged family fit is below the Q2 floor. A target that reaches a positive floor while Q3 is failed is `NON-SPECIFIC / NO POSITIVE CLAIM`, not a candidate.

## 8. Sealed Voynich test

### Folio split

Parse eligible RF1b-er words by page. Sort folios by SHA-256 of `CIPHERCLOSEV1split::<folio>` and split 60/20/20:

- `T1`: first 60% — map fitting only;
- `H1`: next 20% — first target compatibility test;
- `C1`: final 20% — sealed confirmation only if a positive H1 candidate is admissible.

All 16 structural candidates and all eight languages are fitted on T1 before H1 scoring. The complete candidate inventory is retained; no favourable post-hoc subsetting is allowed.

### H1 family decision

For each family M0/TQ/NQ, identify its best converged `(rule, language, map)` by fixed-map H1 score.

A family is **CLOSED NEGATIVE / INCOMPATIBLE UNDER V1** if:

- retained-character coverage >=0.995;
- the best T1 fit converges under A/B criteria;
- its H1 held-out score is below the frozen Q2 5th-percentile positive-control floor for that selected language.

Because the decision uses the *best* member of the full frozen family, multiple rule/language search makes this negative criterion conservative.

A family reaches a positive H1 compatibility floor only if its score is at or above the frozen Q2 floor. This alone is not evidence.

An **H1 candidate** additionally requires:

- Q3 passed;
- candidate score reaches its Q2 floor;
- top-v-runner-up evidence margin >=0.05 nats/character after subtracting each candidate's own Q2 floor;
- A/B map agreement >=0.95;
- same language and rule top in four deterministic H1 folio buckets, with positive evidence above floor in every bucket.

If H1 reaches the positive floor but Q3 or candidate gates fail: `NON-SPECIFIC / NO POSITIVE CLAIM`; C1 remains sealed.

### C1 confirmation

Only an admissible H1 candidate unlocks C1. Apply the T1-fitted map/rule without refitting.

C1 requires:

- same language/rule remains top;
- score >= frozen Q2 floor;
- evidence margin >=0.05;
- positive evidence in four deterministic C1 folio buckets;
- retained coverage >=0.995.

Only after C1 passes may plaintext strings be emitted or inspected.

## 9. Compute discipline

Before every remote launch, list existing Hugging Face jobs. No launch is permitted if an unexplained paid job is running.

- Prefer local/CPU-basic for source census and smoke tests.
- Qualification may use `cpu-xl` only if the bounded CPU-basic smoke is valid.
- Every job must have an explicit timeout <=2 hours; no scheduled or background jobs.
- Failed/stopped branches terminate immediately.
- After every decisive result, list jobs again and cancel anything still running that is no longer scientifically required.
- No orphan paid compute.

## 10. Binding stopping point

- If all surviving families are either CLOSED NEGATIVE or excluded as NON-IDENTIFIABLE/historically inadmissible by Gate 1, issue **BROAD IDENTIFIABLE HISTORICAL CIPHER HYPOTHESIS CLOSED UNDER COVERAGE V1**.
- If a family passes all control, specificity, H1 and C1 gates, issue **TERMINAL CIPHER CANDIDATE <family>/<rule>/<language>** and only then inspect plaintext.
- If an arm cannot qualify or cannot converge on target, preserve **PARTIAL / INSTRUMENT NOT QUALIFIED**. Do not convert lack of solver power into a mechanism negative.

No further cipher-family invention follows this programme without new external historical or manuscript evidence.
