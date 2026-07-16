# Voynich terminal cipher programme v0.6

Date frozen: 2026-07-16

Branch: `experiment/terminal-cipher-programme-v0.6-20260716`

Status: **FROZEN BEFORE NEW DEVELOPMENT RESULTS**

## 1. Purpose

This is the final bounded cryptanalytic programme before reconsidering the cipher hypothesis as a whole.

The programme is not designed to find an attractive plaintext by trying many historical constructions. It is designed to answer a narrower and falsifiable question:

> Given frontier-level modelling, current automatic-decipherment methods and effectively unconstrained parallel compute, do the remaining materially distinct historical cipher mechanisms produce reliable synthetic recovery and then stable, cross-validated evidence on Voynich data?

A comprehensive negative result will not prove that no imaginable cipher could have generated the manuscript. It will, however, remove cipher from the active research hypothesis set and reduce it to a residual possibility requiring new external evidence.

No Voynich text may be scored until the relevant synthetic family has passed development and one untouched locked test.

## 2. Prior completed coverage

The following classes are already closed or validated under the v0.5 programme.

| Primitive or family | Existing result | Consequence |
|---|---|---|
| Fresh monoalphabetic substitution | Strong locked-test recovery | Valid recoverable reference family |
| Fresh-key homophonic substitution | Development and/or locked reliability failure despite classical, beam, recurrence and hybrid arms | No blind Voynich recovery claim permitted |
| Shared known homophone pool | 100% positive control | Reused known symbol systems are materially easier and must not be conflated with fresh-key recovery |
| Nomenclator / opaque whole-word codes | Both codebook and residual-key component gates failed | No joint blind solver permitted |
| Monoalphabetic substitution plus fixed block transposition | Both oracle components passed; final joint development failed | No locked test or Voynich application permitted |

The v0.6 programme therefore excludes new tuning of these closed solvers.

## 3. Remaining material scope

Only mechanisms not already represented in a materially adequate way are in scope.

### Family P — polyalphabetic and stateful substitution

This family represents cipher disks, wheels and changing alphabets.

Synthetic subfamilies:

1. periodic independent substitution alphabets, periods 2–12;
2. line-reset periodic alphabets;
3. irregular alphabet-state changes at bounded intervals of 16–96 characters;
4. signalled and unsignalled state changes;
5. historically constrained Alberti-style alphabet rotations or substitutions;
6. optional short indicator groups, scored separately from unsignalled systems.

The primary form uses full random alphabet permutations rather than only Caesar shifts. Shift-only Vigenère is retained as an easy positive control.

### Family S — syllabic, polygraphic and unsegmented substitution

This family covers systems in which cipher units are not identical to single plaintext characters.

Synthetic subfamilies:

1. fixed digraph substitution;
2. mixed character and common-syllable substitution;
3. variable-length one-to-three-symbol numerical or abstract code groups;
4. omitted or ambiguous separators;
5. bounded code-unit ambiguity and transcription noise;
6. limited whole-word entries layered over the syllabic system, without reopening unrestricted nomenclator development.

### Family T — non-block transposition

Only materially distinct transposition forms are retained:

1. columnar transposition with unknown width 4–16;
2. line-local columnar or route transposition with reset at each line;
3. bounded grille/route patterns that repeat across lines or paragraphs;
4. substitution followed by one of the above, only after the oracle components pass.

Arbitrary unconstrained permutations are excluded as non-identifiable and historically uninformative.

### Family G — bounded steganographic extraction

Steganography is not treated as an unrestricted cipher search. Only independently specifiable carrier channels are permitted:

1. line-initial and line-final glyph channels;
2. fixed positional selections within lines or tokens;
3. odd/even or periodic positions with period at most 12;
4. rare-glyph versus common-glyph binary channels;
5. gallows or other predeclared glyph-class channels;
6. illustration-label versus running-text contrasts where coordinates independently define the carrier.

Every extraction rule must be declared before inspecting decoded output. A rule must transfer across held-out folios and sections. No visual or linguistic reinterpretation after seeing a candidate message is allowed.

## 4. Literature-backed solver portfolio

The implementation must include the strongest distinct approaches supported by current literature, not multiple cosmetic variants of one search.

### Shared components

- historical and modern character n-gram language models;
- neural character language models;
- recurrence-canonical encodings;
- whole-sequence neural scoring;
- simulated annealing or hill climbing;
- exhaustive enumeration where the structural state space is bounded;
- lattice decoding for ambiguous segmentation;
- calibrated family detection and abstention.

### Family P portfolio

1. classical period and state-change diagnostics: index of coincidence, lagged mutual information, Kasiski-style repetition and Bayesian change points;
2. exact or beam enumeration of short period/state schedules;
3. column-conditioned substitution solvers using historical language models;
4. neural sequence models trained on arbitrary fresh alphabet states;
5. hybrid neural proposals plus classical key refinement;
6. full-sequence language-model reranking.

### Family S portfolio

1. unigram and BPE segmentation baselines;
2. probabilistic lattice segmentation and decoding;
3. recurrence-aware sequence models;
4. masked language-model assignment for syllabic/code units;
5. neural-LM lattice rescoring;
6. classical joint segmentation/key search where bounded.

### Family T portfolio

1. exact width and route enumeration where feasible;
2. language-model scoring of detransposed streams;
3. coordinate search over transposition and substitution;
4. neural proposal models only if they demonstrate independent development gain.

### Family G portfolio

1. exact extraction enumeration within each frozen channel family;
2. multiple-testing-corrected language-model and compression scores;
3. held-out transfer testing;
4. matched random-carrier and structured-generator controls.

## 5. Corpora and data partitions

### Synthetic plaintexts

The six pinned v0.5 UD corpora remain the reproducible multilingual base:

- English;
- German;
- Finnish;
- Turkish;
- Hebrew;
- Arabic.

Before family development, Stage 0 must attempt to add pinned historical Latin and northern/central Italian corpora. These additions are accepted only if licensing, provenance and deterministic preprocessing are documented. Failure to obtain adequate historical corpora must be recorded rather than silently replaced with modern approximations.

### Lengths

Each family is evaluated at lengths 192, 384, 768 and 1,536 characters where structurally meaningful. Voynich-scale line and page aggregates are included separately.

### Splits

- train: model fitting and synthetic example generation;
- development: architecture and one permitted amendment;
- locked test: untouched family validation;
- sealed Voynich: unavailable until the family locked test passes.

No plaintext chunk, key, state schedule or generated ciphertext may cross partitions.

## 6. Family gates

Each subfamily has an oracle-component stage before joint solving.

### Oracle gates

Every independently hidden component must achieve:

- mean character or unit recovery at least 90%;
- median at least 95%;
- at least 18/20 trials at or above 80%;
- no catastrophic failure below 50% in more than 1/20 trials.

If an oracle component fails after one permitted development amendment, the joint family is blocked.

### Joint development gate

For each language-length cell with 20 trials:

- mean plaintext recovery at least 75%;
- median at least 90%;
- at least 16/20 trials at or above 70%;
- structural parameter accuracy at least 80%;
- score-selected solution must outperform the frequency or identity baseline by a preregistered margin;
- no use of the true key, language label or family label beyond the declared condition.

Aggregate success cannot conceal a failed language or structural regime. A family is only admitted for locked testing in the cells that individually pass.

### Locked test gate

The selected development configuration is run once on new keys and plaintext blocks.

Required:

- mean recovery at least 75%;
- median at least 90%;
- at least 16/20 trials at or above 70%;
- structural parameter accuracy at least 80%;
- no post-test modification.

A locked-test failure closes that family.

## 7. Blind model-selection layer

Passing recovery alone is insufficient. A separate classifier and evidence layer must distinguish:

- monoalphabetic substitution;
- polyalphabetic/stateful substitution;
- syllabic/polygraphic substitution;
- bounded transposition;
- mixed historical family;
- structured generated text;
- abbreviated or conventional notation;
- ordinary language-like text;
- none of the represented families.

Inputs must be label-invariant and include:

- recurrence patterns;
- collision spectra;
- lagged dependence;
- conditional entropy;
- token and line boundary effects;
- section and hand stability;
- segmentation uncertainty;
- compression and minimum-description-length terms;
- recovery stability across independent solver runs.

The layer must be calibrated on matched negative controls and must support abstention.

Required locked synthetic performance:

- macro-AUC at least 0.90;
- expected calibration error at most 0.05;
- false-positive rate on structured generated controls at most 5% at the declared evidence threshold;
- abstention permitted and scored as correct on out-of-family controls.

## 8. Voynich application protocol

A family may be applied only if its synthetic locked test passes.

The application must be blind to interpretive labels and use at least three independently prepared representations where available, including line-aware recurrence encodings.

A positive Voynich result requires all of the following:

1. the same family is selected across representations;
2. evidence survives held-out folios and sections;
3. recovered structural parameters transfer across at least two independent manuscript partitions;
4. independent solver restarts converge on equivalent keys or plaintext structure;
5. recovered text or units improve held-out language-model likelihood, not only training likelihood;
6. the result beats matched generated-text and notation controls under the frozen evidence rule;
7. no manual insertion, word guessing or semantic selection is used;
8. any proposed plaintext is auditable character by character from a frozen key or decoding lattice.

A visually or semantically attractive isolated output is not evidence.

## 9. Terminal decision rule

The programme ends after Families P, S, T and G and the blind model-selection layer. No new family may be added merely because the results are negative.

### Cipher remains active only if

At least one family:

- passes its synthetic locked recovery gate;
- produces stable, held-out, cross-representation evidence on Voynich;
- yields transferable structural parameters or an auditable partial plaintext;
- beats structured generation and notation controls.

### Cipher is removed from the active hypothesis set if

- no represented family produces qualifying Voynich evidence; or
- all families capable of passing synthetic recovery classify Voynich as out of family or abstain; and
- bounded steganographic channels yield no held-out transferable signal after multiple-testing correction.

The conclusion will be stated as:

> No historically material cipher mechanism represented by the terminal programme produced recoverable, stable and cross-validated evidence on the Voynich Manuscript. Cipher is therefore no longer retained as a working explanation, although logically unconstrained or externally anchored constructions cannot be disproved.

No numerical posterior is fixed in advance. The final posterior will be calculated from calibrated synthetic false-positive and false-negative rates, with sensitivity analysis.

## 10. Amendment and stopping discipline

- one development amendment per family;
- no locked-test tuning;
- no threshold relaxation;
- no manual plaintext fishing;
- no unregistered extraction rule;
- no family proliferation after negative results;
- failed branches and implementation defects remain documented;
- invalid runs are replaced only when the defect is demonstrated independently of the outcome;
- all scientific outputs receive SHA-256 hashes and immutable job identifiers.

## 11. Execution order

1. Stage 0: literature, historical-corpus and existing-code audit.
2. Family P oracle grid, joint development and locked test if permitted.
3. Family S oracle grid, joint development and locked test if permitted.
4. Family T oracle grid, joint development and locked test if permitted.
5. Family G bounded extraction and held-out controls.
6. Blind model-selection layer.
7. Sealed Voynich application of only locked-passing families.
8. Final comparative report and posterior reassessment.

Families P, S and T may execute in parallel after Stage 0 freezes their manifests. Family G can execute in parallel because it does not depend on a plaintext recovery solver.

## 12. Initial literature anchors

The Stage 0 audit begins from, but is not limited to:

- Kambhatla, Bigvand and Sarkar, *Decipherment of Substitution Ciphers with Neural Language Models* (2018);
- Aldarrab and May, *Can Sequence-to-Sequence Models Crack Substitution Ciphers?* (2020);
- Chu, Valenti and Knight, *Solving Historical Dictionary Codes with a Neural Language Model* (2020);
- Aldarrab and May, *Segmenting Numerical Substitution Ciphers* (2022);
- Kambhatla, Born and Sarkar, *Decipherment as Regression* (2023);
- Megyesi et al., *Historical Language Models in Cryptanalysis* (2023);
- Bruton, Beloucif and Megyesi, *Attention-Augmented LSTMs for Automatic Homophonic Ciphertext Decipherment* (2026);
- CrypTool 2 historical-cipher implementations and pinned source provenance.

Stage 0 must search direct ArXiv records, peer-reviewed indexes, HistoCrypt proceedings, Cryptologia, ACL/EMNLP and current open-source solvers before implementation begins.
