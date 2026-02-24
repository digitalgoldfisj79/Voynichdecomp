# S3: Semantic Tests — Positive Controls and Negative Results

> **Source code:** `src/p70c_connect.py` (suffix decomposition validation)  
> **Data:** `data/enriched_records.pkl` (37,465 decomposed tokens)


## S3.1 Methodology: Positive-Controlled Falsification

Each semantic test follows a two-stage design:

1. **Validation stage**: Apply the method to a known medieval manuscript where semantic content is recoverable. If the method fails the positive control, it cannot be used to draw conclusions about the VMS.
2. **Application stage**: Apply the identical method to the VMS. If the method passes the positive control but fails on VMS, this constitutes evidence against semantic content in the VMS (under the detection model validated by the control).

## S3.2 Test 1: Section Classification Under Shuffling

**Positive control**: N/A (internal test).

**Method**: Train a Naive Bayes classifier on token frequencies to distinguish the 5 major VMS sections. Then shuffle all token order within each folio and retrain.

**Results**:

| Condition | Accuracy | 95% CI |
|-----------|----------|--------|
| Original order | 76–81% | ±3% |
| Shuffled order | 76–81% | ±3% |
| Permutation baseline | 22% | ±2% |

**Interpretation**: Classification accuracy is identical under shuffling. The section signal is entirely bag-of-words (vocabulary composition), not syntactic. This reframes the finding of Bowern and Lindemann (2021): the vocabulary differentiation is real, but the interpretation as semantic must be revised. Section-specific vocabulary reflects notation conventions rather than topic encoding.

## S3.3 Test 2: Word-Order Sensitivity (Bigram Folio Similarity)

**Positive control**: *Tacuinum Sanitatis* (genuinely formulaic medieval pharmaceutical text).
- MI₁ = 1.79 bits
- Shuffling degrades folio similarity by 52%
- Frame ablation degrades by 99%

**Method**: Compute pairwise folio similarity using bigram frequency vectors. Compare original vs within-folio shuffled similarity matrices (Spearman ρ between the two).

**Results**:

| Corpus | ρ(original, shuffled) | Degradation |
|--------|----------------------|-------------|
| Tacuinum Sanitatis | 0.15 | 52% |
| VMS | 0.32 | 0% |

The VMS ρ = 0.32 is entirely predicted by vocabulary overlap: Jaccard = 0.19 predicts ρ = 0.31 in simulation. No additional sequential information contributes.

## S3.4 Test 3: External Groupings (14 Categories)

**Positive control**: Florence, Biblioteca Riccardiana MS 106 (15th-century Italian herbal). Categories with genuine therapeutic vocabulary show elevated Jaccard similarity (p < 0.01 by permutation test).

**Method**: For each of 14 external groupings derived from published identifications, compute within-group lexical coherence (mean pairwise Jaccard similarity of folio vocabulary). Test significance against a null distribution from 10,000 random groupings of equal size.

**External groupings tested** (all specified a priori):

| Source | Categories | n folios |
|--------|-----------|----------|
| Tucker & Janick (2016) plant IDs | 6 families | 28 |
| Galenic humoral classification | 4 categories | 42 |
| Illustration-based grouping | 4 types | 38 |
| Therapeutic class (published) | 4 categories | 32 |

**Results**: 0 of 14 groupings survive Bonferroni correction at α = 0.0063. The highest observed p-value is p = 0.023 (uncorrected), which does not survive multiple-comparison correction.

## S3.5 Test 4: Same-Plant Falsification (f48v vs f89v2)

**Method**: Folios f48v and f89v2 have been identified in the literature as depicting the same plant species. If the text describes illustrations, these folios should share content vocabulary.

**Results**:

| Metric | Value |
|--------|-------|
| Shared content cores | 0 |
| Jaccard (content cores) | 0.000 |
| Shared structural cores | 4 |
| Shared structural cores (all-section) | 4/4 (100%) |

All shared vocabulary consists of structural cores that appear uniformly across the manuscript. Zero content-specific cores are shared. This single-case falsification tests the strongest version of the "text describes illustrations" hypothesis.

## S3.6 Test 5: Word-to-Word Mutual Information

**Positive control**: Natural language texts in Latin, Italian, German, English.
- MI₁ ranges from 3–5 bits
- Sequential structure is dominated by lexical association

**Method**: Compute MI between adjacent words (bigram MI₁). Decompose into frequency component (MI predicted by marginal frequencies alone) and sequential component (residual).

**Results**:

| Corpus | MI₁ (bits) | % from frequency | % sequential |
|--------|-----------|-------------------|-------------|
| VMS | 0.45 | 98% | 2% |
| Latin (Tacuinum) | 3.21 | 41% | 59% |
| Italian (Riccardiana) | 2.87 | 38% | 62% |

VMS word-to-word MI is an order of magnitude below natural language. The 98% frequency component means that virtually all predictability between adjacent words comes from how common each word is, not from sequential association. The 2% sequential component is entirely explained by the suffix-to-prefix transition grammar documented in §3.2 of the main paper.
