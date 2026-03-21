# Supplement S6a: Greek as Alternative Source Language

**Addendum to S6 (Language Discrimination Battery)**

Edward Bozzard · ORCID 0009-0002-4052-0994

Date: 21 March 2026

---

## Motivation

The language discrimination battery (S6) identified pharmaceutical Latin as the best-fit source language (χ² = 0.458), with Greek flagged as undertested (N = 465, preface only). Subsequent investigation of the Byzantine/Veneto Greek scholarly milieu — Cardinal Bessarion's Marciana collection, the Dioscorides manuscript tradition, Vaticanus Graecus 1291 zodiac parallels, and the Greek-speaking community in fifteenth-century Venice and Padua — provided independent historical motivation to test Greek rigorously as an alternative source language.

This addendum reports three progressively stronger tests of Greek pharmaceutical text against the confirmed cipher architecture, culminating in a head-to-head comparison of the same foundational text (Dioscorides, *De Materia Medica*) in both languages.

## Corpora

Six Greek pharmaceutical corpora were assembled from the OpenGreekAndLatin/First1KGreek repository and the Wellmann critical edition (Kühn reprint):

| Corpus | TLG | Words | Genre |
|--------|-----|------:|-------|
| Galen, *De simplicium medicamentorum* | 0057.077 | 101,152 | Theoretical pharmacology |
| Galen, *De comp. med. per genera* | 0057.078 | 27,185 | Drug recipes by type |
| Galen, *De comp. med. sec. locos* | 0057.079 | 12,804 | Drug recipes by body part |
| Dioscorides, *Euporista* | 0530.006 | 9,112 | Simple remedies |
| Hippocrates, *De natura muliebri* | 0627.019 | 3,791 | Gynaecological recipes |
| Hippocrates, *De morbis* | 0627.026 | 8,125 | Medical prose |
| **Dioscorides, *De Materia Medica* I–V** | **0530.001** | **111,825** | **Comprehensive materia medica** |

The Dioscorides DMM was extracted from the Wellmann/Kühn edition (Greek text, 111,825 romanised words, 25,257 types). A Latin token list from the same edition (133,224 tokens) was also tested but proved contaminated with editorial apparatus and is excluded from the main comparison.

All Greek text was romanised using standard scholarly transliteration (α→a, β→b, γ→g, δ→d, θ→th, φ→ph, χ→ch, ψ→ps, ξ→x) before processing through the confirmed cipher architecture. No Greek-specific adaptations were made to the grid, nomenclator, or any other cipher component.

## Method

### Test 1: Screening battery (m_core leading character distribution)

The S6 battery computes χ² between source-language first-consonant group distributions and the VMS Herbal-A m_core leading character distribution. This was extended to include combined Greek pharmaceutical text (122,180 words) with and without a function word filter.

The function word filter removes Greek articles (τό, τήν, τοῦ, etc.), particles (δέ, τε, γάρ, μέν), and prepositions (ἐν, ἐκ, εἰς, ἐπί, etc.) — 33.2% of running text — on the grounds that the two-table cipher architecture routes function words through the nomenclator (EC layer), not the grid (FC layer). Under this architecture, function word consonants do not appear in the m_core distribution.

### Test 2: Forward cipher (84-metric scoring suite)

The confirmed clean-baseline forward cipher (S4) was run with Greek input using identical parameters: same grid (ROW_MAP), same VMS cell pools (Herbal-A), same EC threshold (53%), same scoring suite (S5, 90 metrics with v2 tolerances). Five seeds (42, 404, 501, 606, 808) were used for each configuration.

A Greek nomenclator was constructed by functional analogy to the confirmed Latin nomenclator: καί→Y (parallel to *et*→Y), ἐν→N (*in*→N), ἀπό→L (*de*→L), etc. — 12 entries matching the 12 Latin entries.

### Test 3: Component-level discrimination

Three component-level tests were designed to exploit structural differences between Greek and Latin that survive the cipher transform:

**Suffix family distribution.** The confirmed mapping V₁→suffix family (a→Y, e→R, i→N, o→L, u→BARE) predicts suffix family proportions from source-language first-vowel distributions. This test has zero free parameters.

**C₁–C₁ bigram correlation.** Adjacent content-word first consonants in the source text produce m_core leading-character bigrams in the cipher output. The Pearson correlation between predicted and observed bigram distributions measures how well each source language reproduces VMS word-adjacency patterns.

**Source word length.** Mean word length of content words in each language was compared to VMS token lengths.

## Results

### Screening battery

| Corpus | N | χ² | Status |
|--------|--:|---:|--------|
| CI Latin (confirmed baseline) | 24,300 | 0.458 | S6 result |
| Greek pharma (all words) | 122,180 | 0.576 | Competitive |
| Greek pharma (no function words) | 81,649 | 0.138 | Lowest χ² |

The function word filter reduces χ² from 0.576 to 0.138, apparently beating CI Latin. However, this comparison is invalid: the Latin battery score (0.458) was computed without an equivalent filter, and the Greek filter introduces three free parameters (function word list, row mapping for Greek consonants, classification of aspirated stops). The screening battery cannot reliably discriminate between Greek and Latin pharmaceutical text.

### Forward cipher (aggregate score)

| Configuration | Mean score (5 seeds) | C15 | Types |
|---------------|--------------------:|----:|------:|
| CI Latin + Latin nomenclator | 51.4/90 | 9.0 | 872 |
| Greek DMM (no nomenclator) | 52.2/90 | 9.6 | 800 |
| Greek DMM + Greek nomenclator | 52.2/90 | 10.4 | 807 |
| Greek recipe corpus (Galen) | 48.6/90 | 9.4 | 747 |

The Dioscorides DMM in Greek scores 52.2/90, marginally above CI Latin's 51.4/90. Individual seed ranges overlap completely (Latin 49–53, Greek 50–55). The forward cipher aggregate score does not discriminate between the two languages at the clean-baseline level.

Greek produces fewer types (800–807 vs 872; VMS target 1430), reflecting greater consonant collision in the grid: Greek κ, γ, ξ, and ψ all route to row 'o', reducing output diversity.

### Component-level discrimination

**Suffix family distribution (zero free parameters):**

| | Y | N | L | R | BARE | χ² vs VMS |
|---|---:|---:|---:|---:|-----:|----------:|
| VMS (observed) | 28.0% | 14.5% | 12.6% | 16.7% | 21.9% | — |
| CI Latin (predicted) | 25.6% | 18.8% | 19.4% | 20.4% | 15.8% | **0.078** |
| Greek DMM (predicted) | 26.7% | 11.5% | 20.3% | 31.1% | 10.4% | 0.240 |

Latin's predicted suffix family distribution matches VMS 3.1× better than Greek's (χ² = 0.078 vs 0.240). The critical mismatch for Greek is the R-family: Greek predicts 31.1% (driven by the high frequency of ε-initial words, since both η and ε romanise to 'e') against VMS 16.7%. Greek also underpredicts BARE (10.4% vs 21.9%), because υ-initial words are rare in Greek. These mismatches are structural properties of the language and are not correctable by corpus selection.

**C₁–C₁ bigram correlation:**

| Source | Pearson r | p-value |
|--------|----------:|--------:|
| CI Latin | **0.9728** | 1.75 × 10⁻⁵⁴ |
| Greek DMM | 0.9419 | 4.46 × 10⁻⁴¹ |

Latin's consonant adjacency patterns correlate with VMS 0.031 points higher than Greek's. Both correlations are strong (>0.94), but Latin is consistently closer to the independently validated r = 0.96 from the Ald.211 held-out test (Paper 2, §5.5).

**Source word length:**

| Corpus | Mean | Median |
|--------|-----:|-------:|
| CI Latin content words | 6.32 | 6.0 |
| Greek DMM content words | 7.03 | 7.0 |
| VMS Herbal-A tokens | 4.64 | 5.0 |

Neither source language's word length matches VMS output directly (the cipher compresses), but Latin is closer. Greek's longer mean word length (driven by polysyllabic compound forms) would require systematic truncation to produce VMS-length output.

### Cross-section validation

All three component-level tests were repeated across all six VMS sections. Latin wins 16 of 18 section × test comparisons; Greek wins 2 (Balneological only, on bigram correlation and row distribution).

**Suffix family χ² (Latin wins 6/6):**

| Section | N_FC | Latin χ² | Greek χ² | Ratio |
|---------|-----:|--------:|---------:|------:|
| Herbal-A | 1,737 | 0.078 | 0.240 | 3.1× |
| Herbal-B | 2,538 | 0.189 | 0.390 | 2.1× |
| Pharmaceutical | 2,151 | 0.093 | 0.211 | 2.3× |
| Stars | 5,340 | 0.116 | 0.255 | 2.2× |
| Balneological | 2,501 | 0.523 | 0.672 | 1.3× |
| Zodiac | 1,006 | 0.326 | 0.350 | 1.1× |

Latin wins every section. The margin narrows for Balneological and Zodiac (where neither language fits well), but Latin is never worse.

**C₁–C₁ bigram correlation (Latin wins 5/6):**

| Section | Latin r | Greek r | Δ |
|---------|--------:|--------:|---:|
| Herbal-A | 0.973 | 0.942 | +0.031 |
| Herbal-B | 0.881 | 0.817 | +0.064 |
| Pharmaceutical | 0.416 | 0.288 | +0.128 |
| Stars | 0.310 | 0.286 | +0.024 |
| Balneological | 0.059 | 0.075 | −0.016 |
| Zodiac | 0.174 | 0.074 | +0.101 |

Latin wins 5 of 6 sections. The single Greek win (Balneological, Δ = −0.016) is within noise. The largest Latin advantage is Pharmaceutical (+0.128), consistent with pharmaceutical Latin being the best-fit source genre.

Note that bigram correlations decline sharply from Herbal-A (0.97) through non-herbal sections (<0.42), consistent with the herbal sections being closest to pharmaceutical source text and the other sections encoding different content types.

**Row distribution χ² (Latin wins 5/6):**

| Section | Latin χ² | Greek χ² | Ratio |
|---------|--------:|---------:|------:|
| Herbal-A | 0.021 | 0.078 | 3.7× |
| Herbal-B | 0.143 | 0.219 | 1.5× |
| Pharmaceutical | 0.677 | 1.021 | 1.5× |
| Stars | 1.138 | 1.187 | 1.0× |
| Balneological | 2.100 | 1.908 | 0.9× |
| Zodiac | 2.224 | 2.581 | 1.2× |

Latin wins 5 of 6. Greek's sole win (Balneological, ratio 0.9×) is marginal.

## Discussion

The forward cipher architecture does not discriminate between Greek and Latin pharmaceutical text at the aggregate scoring level. Both languages score 51–52/90 through the same confirmed grid with no parameter tuning. This near-equivalence likely reflects the high proportion of Greek loanwords in Latin pharmaceutical vocabulary (~40% of CI terms are Greek-derived), which ensures similar consonant-group distributions in both languages for this genre.

Discrimination emerges at the component level. The suffix family test — which has zero free parameters, since the V₁→family mapping is confirmed architecture — separates the two languages cleanly: Latin χ² = 0.078, Greek χ² = 0.240. This test exploits a genuine structural difference: Greek's vowel inventory maps disproportionately to the R-family (via η/ε → 'e') and away from BARE (via rare υ), while Latin's more even first-vowel distribution matches the VMS suffix family proportions three times better.

The bigram correlation reinforces this finding: Latin r = 0.973 vs Greek r = 0.942. The Greek nomenclator, constructed by functional analogy to the validated Latin nomenclator, does not improve the Greek score and slightly worsens it on some configurations — suggesting that Greek function words do not map to VMS suffix families with the same precision as Latin function words.

These results do not exclude Greek-influenced Latin (e.g., a Latin translation or adaptation of a Greek pharmaceutical source) as the VMS plaintext. The cipher architecture would produce identical output for a Greek loanword regardless of whether it entered via Greek or Latin. What the results exclude is **untranslated Greek** as the direct cipher input: the first-vowel distribution of Greek pharmaceutical prose is structurally incompatible with the VMS suffix family proportions.

## Conclusion

Greek pharmaceutical text is competitive with Latin on aggregate cipher-output statistics but is discriminated against by component-level tests across all six VMS sections. Latin wins 16 of 18 section × test comparisons (suffix family 6/6, bigram correlation 5/6, row distribution 5/6). Greek's two wins are both in the Balneological section, by margins within noise. The identification of pharmaceutical Latin as the source language — established in S6 and validated by the r = 0.96 bigram result in Paper 2 — survives the strongest available alternative hypothesis tested against the most plausible alternative corpus (the foundational pharmaceutical text of the ancient world in both languages).

The Byzantine/Veneto Greek milieu remains relevant to the production context (Paper 3) regardless of source language, since production environment and content language are independent variables.

## Data availability

All Greek corpora, processing code, and scoring results are archived at:
- `greek_battery_v2_2.py` — battery and forward cipher code
- `greek_corpus_parsed.pkl` — romanised Greek pharmaceutical corpus
- `greek_dmm_corpus.pkl` — romanised Dioscorides DMM
- `dmm_definitive_results.pkl` — forward cipher scoring results

Source texts from OpenGreekAndLatin/First1KGreek (MIT licence) and the Wellmann/Kühn edition (public domain).
