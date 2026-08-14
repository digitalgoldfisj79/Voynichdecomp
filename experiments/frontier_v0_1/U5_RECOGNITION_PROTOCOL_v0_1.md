# U5-B — blind fresh-codebook verbose-family recognition protocol v0.1

Date frozen: 2026-08-14
Status: **FROZEN BEFORE U5-B DEVELOPMENT/CALIBRATION GENERATION**
Voynich status: **SEALED**
Prerequisite: U5-A `PASS_RECOVERY_CALIBRATION` (mean 0.98867; 20/20 ≥0.75).

## Question

Can the Naibbe-like *mechanism class* be recognised from an unseen ciphertext surface when literal codewords, plaintext source, and hidden key are all unseen?

A recogniser trained on Greshko's literal Voynich-like codewords would be circular for application to the Voynich Manuscript. U5-B therefore forbids literal Naibbe glyph/token identities from entering the classifier. Every positive sample uses a **fresh independently generated codebook over an artificial alphabet**, retaining only the abstract Naibbe architecture and empirical role-length distributions.

Passing U5-B is required before any Voynich score is permitted. U5-A recovery alone is insufficient.

## Positive family: FRESH-VERBOSE

For every sample independently:

1. Normalize the plaintext to Naibbe's 23-letter alphabet (W→UU, J→I, K→C; diacritics removed).
2. Draw one fresh global permutation of the 23 plaintext identities.
3. Generate new unigram, prefix and suffix codebooks from scratch over the artificial atomic alphabet `abcdefghijklmnopqrst`.
4. For every plaintext identity and each of six states, generate a unique role codeword. Codeword lengths are sampled from the corresponding empirical Naibbe unigram/prefix/suffix **length distribution only**. The literal Greshko codewords are never used as features or positive surfaces.
5. Segment plaintext into one- or two-character units with P(1)=P(2)=0.5, subject to the terminal character.
6. Singles choose the Naibbe unigram state weights; pairs choose prefix and suffix states independently and concatenate the two fresh role codewords. Pair strings colliding with fresh unigram strings are rejected and redrawn.

Every sample has a new codebook. No codeword string is shared intentionally between development and locked test samples.

## Source split

Development/calibration plaintext sources:

- Latin: Caesar, *De Bello Gallico* I–IV, Project Gutenberg #218.
- Italian: Collodi, *Le avventure di Pinocchio*, Project Gutenberg #52484.

Locked recognition sources, never used for classifier fitting or threshold selection:

- Latin: Pliny, *Naturalis Historia* XVI from the Naibbe repository.
- Italian: Dante, *Divina Commedia* from the Naibbe repository.

Thus both plaintext source/author and codebook are held out.

## Sample size

Each sample is generated from **4,096 normalized plaintext characters**. Development produces 120 positives (60/source-language); 80 positives are used to fit the classifier and 40 for threshold calibration. Locked testing produces 100 positives (50/source-language).

## Matched negative families

For each positive sample, four negatives are generated, yielding a fixed 1:4 positive:negative operational prevalence.

### N1 — TYPE-RECODE

Preserve the exact positive token sequence's type-frequency spectrum and every token length, but replace every distinct token type by a fresh random string of the same length. This destroys reusable internal prefix/suffix construction while preserving exact repetition, vocabulary size, token order and length.

### N2 — GLOBAL-CHAR-RESHUFFLE

Preserve the positive sample's exact atomic-character multiset and exact token-length sequence, globally permute all atomic characters, then repartition by the original token lengths. This preserves symbol inventory, unigram mass and token lengths while destroying codeword structure.

### N3 — TOKEN-INTERNAL-SHUFFLE

Independently permute the characters inside every positive token. Token order, token lengths and each token's character multiset are preserved; internal role order is destroyed.

### N4 — DEPENDENT-SLOT3

A structured non-message/morphology control. Preserve the positive token-length sequence exactly, but generate each token from reusable prefix/core/suffix dictionaries over the same artificial alphabet. For lengths ≥3, all three components are present; length 2 uses two components; length 1 one. Component choices are deterministically conditioned on adjacent plaintext identities so the control has reusable slots **and dependency**, not merely IID morphology. This is the principal anti-confounding null: a recogniser that merely detects compositional token morphology should fail here.

## Representation firewall

Before feature extraction, every sample's atomic characters are relabelled canonically in order of first appearance. No feature may use original character labels, literal token identities across samples, Greshko table names, plaintext identity, language, author or source.

Token lengths are audited but **not used as classifier features**, because the positive length distributions ultimately derive from a Voynich-inspired construction.

## Frozen feature vector

Only equality/reuse/order structure within a sample is admitted:

1. type/token ratio;
2. hapax share among types;
3. repeated-occurrence share;
4–6. occurrence-weighted distinct-type prefix support at lengths 1,2,3;
7–9. analogous suffix support;
10. mean best bipartite split support: for each token, maximise `min(prefix_support, suffix_support)` over all internal cuts and average `log1p` support;
11. fraction of token occurrences with some cut where both component supports are ≥3;
12. same at support ≥5;
13. best-split reusable dictionary character-cost / flat type-dictionary character-cost;
14. best-split bipartite 4-cycle/rectangle closure density;
15. mean entropy of suffix partners given inferred prefix;
16. mean entropy of prefix partners given inferred suffix;
17. within-token adjacent-character equality rate;
18. token-boundary adjacent-character equality rate;
19. within-token character bigram mutual information after canonical relabelling;
20. token-boundary character mutual information;
21. mean longest-common-prefix fraction to the best other token type;
22. mean longest-common-suffix fraction to the best other token type.

No surface n-gram identity, codeword dictionary lookup, word/string embedding or language-model score is allowed.

## Classifier and threshold

- StandardScaler + L2 logistic regression, `C=1`, class_weight=`balanced`, solver=`liblinear`, random_state=20260814.
- Fit on the fixed 80 development positives plus their 320 matched negatives.
- Threshold chosen **once** on the remaining 40 development positives +160 negatives. Among thresholds realised by calibration probabilities, select the threshold with greatest recall subject to precision ≥0.95 and **every negative-family FPR ≤0.05**. Ties choose the higher threshold. If none exists with nonzero recall, recognition fails without opening the locked test for threshold selection.
- The selected classifier and threshold are then frozen and evaluated once on 100 locked positives +400 matched negatives.

## Formal U5-B gate

PASS only if the locked test simultaneously has:

- recall ≥ **0.80**;
- precision ≥ **0.95**;
- **each** matched negative-family FPR ≤ **0.05**.

The per-family reading is the conservative operationalisation of the umbrella `matched_null_fpr_max=0.05`; aggregate FPR is also reported.

If recovery passed but recognition fails, the U5 conclusion is **`RECOVERABLE_NOT_IDENTIFIABLE`** and Voynich remains sealed.

## Target firewall

No Voynich corpus, statistic, token inventory, transliteration or generator output is read while fitting, calibrating or testing U5-B. A Voynich score may be computed only after the complete locked U5-B gate passes. Even then it is a mechanism-compatibility score, not a decipherment or provenance result.