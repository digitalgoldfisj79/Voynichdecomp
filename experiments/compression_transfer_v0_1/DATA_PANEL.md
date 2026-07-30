# Registered data panel and acquisition plan

## 1. General eligibility

Every document must have:

- stable source URL or archive identifier;
- licence permitting local scientific processing and, where possible, redistribution;
- exact downloaded-file SHA-256;
- document, work and author identifiers;
- language/family label fixed independently of text statistics;
- documented normalization from source to analysis text;
- duplicate and overlap screening.

A corpus with uncertain identity may be included as an out-of-family control, not as labelled ground truth.

## 2. Stage 1 known-source panel

Minimum eight classes:

1. medieval/early-modern Latin;
2. English;
3. German;
4. Finnish;
5. Turkish;
6. Greek;
7. Hebrew;
8. Arabic.

Expansion candidates may be added before freezing: Italian, French, Spanish, Czech, Dutch and Middle High German. The final list is frozen before development analysis.

Each class requires at least 12 documents, at least two works, and where possible at least two authors. Splits are at document or work level. Parallel translations and duplicated editions may not cross splits.

Two panels are retained:

- **period-tolerant panel:** language recognition across mixed dates and genres;
- **historical-domain panel:** medieval/early-modern material only.

The historical-domain result is the relevant source-transfer calibration for Voynich. Modern-text performance cannot substitute for it.

## 3. Stage 2 synthetic cipher panel

Use existing repository generators and validated implementations where available. Every locked item uses:

- unseen source document;
- fresh independently sampled key;
- fresh renderer parameters;
- no key or codebook sharing with training;
- exact plaintext and key retained for scoring but unavailable to the distance method.

Required families:

- monoalphabetic substitution;
- bounded homophonic substitution;
- nomenclator-style character plus whole-word codes;
- substitution plus block transposition;
- periodic/changing-alphabet Family P;
- at least one null-bearing family;
- at least one polygraphic or fractionating family.

Shared-key and shared-code-pool variants are positive controls only.

## 4. Stage 2 non-cipher controls

Required:

- repository section-conditioned generators;
- generator-disjoint held-out generator configurations;
- human-produced meaningless writing;
- notation-like sequences;
- CoReMA procedural corpus at the representation level permitted by its licence;
- *Polygraphia*-style fixed-position or table-generated material already represented in the repository;
- unigram, token and block shuffles;
- Markov and motif controls trained on training partitions only;
- ordinary prose.

The same source text should be rendered through multiple families to separate source from renderer effects.

## 5. Sealed Voynich panel

The sealed manifest will specify:

- exact transcription release and checksum;
- folio boundaries;
- Currier and section labels used only for stratification;
- uncertain-glyph policy;
- line and word separator policy;
- all preregistered representations;
- exclusion rules based only on missingness or transcription integrity.

No target reference corpus may be added after the first Voynich distance matrix is produced.

## 6. Acquisition stop conditions

Do not run formal Stage 1 when:

- any class has fewer than 12 eligible documents;
- author/work leakage remains unresolved;
- a licence or provenance field is blank;
- class lengths cannot support the registered probe geometry;
- duplicate screening has not been recorded.

Do not run Stage 2 when fresh-key provenance cannot be certified.
