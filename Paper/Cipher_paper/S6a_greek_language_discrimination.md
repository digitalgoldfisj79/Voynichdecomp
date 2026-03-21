# Supplement S6a: Source Language Discrimination — Greek, Mixed Latin-Italian, and the Limits of Aggregate Scoring

**Addendum to S6 (Language Discrimination Battery)**

Edward Bozzard · ORCID 0009-0002-4052-0994

Date: 21 March 2026

---

## 1. Motivation

The language discrimination battery (S6) identified pharmaceutical Latin as the best-fit source language (χ² = 0.458), with Greek flagged as undertested (N = 465, preface only). Two lines of investigation motivated the extended testing reported here.

First, the Byzantine/Veneto Greek scholarly milieu — Cardinal Bessarion's Marciana collection (482 Greek MSS donated to Venice 1468), the Dioscorides manuscript tradition, Vaticanus Graecus 1291 zodiac parallels, and the Greek-speaking community in fifteenth-century Venice and Padua — provided independent historical motivation to test Greek rigorously as an alternative source language.

Second, the observation that the forward cipher v11 aggregate score validates the cipher architecture and scribe production model but may not discriminate between source texts (§5 below) required the development of component-level tests that bypass the scribe layer entirely.

## 2. Greek Corpora

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

The Dioscorides DMM was extracted from the Wellmann/Kühn edition (Greek text, 111,825 romanised words, 25,257 types). All Greek text was romanised using standard scholarly transliteration (α→a, β→b, γ→g, δ→d, θ→th, φ→ph, χ→ch, ψ→ps, ξ→x) before processing through the confirmed cipher architecture. No Greek-specific adaptations were made to the grid, nomenclator, or any other cipher component.

## 3. Mixed Latin-Italian Corpus

Wellcome Collection MS.208 is a collection of medical recipes and pharmacological compounds, mostly in Italian but also in Latin, from late fifteenth-century Padua. The catalogue identifies the main scribe as probably active at the Benedictine abbey of Santa Giustina in Padua, based on an internal reference. The manuscript's watermarks date to 1486 (Cittadella, north of Padua). The spelling is consistent with the Veneto vernacular. Although the watermarks postdate the VMS vellum (radiocarbon 1404–1438), radiocarbon dates the animal skin, not the writing event. Stockpiled parchment was standard practice. Wellcome 208 is tested as a genre exemplar, not as a candidate source text.

The manuscript was transcribed using Transkribus Text Titan I ter (model ID 356425, v2.42.0, run 21 March 2026). Yield: 46,943 words across 246 pages.

Three segments were defined following the Wellcome catalogue:

| Segment | Folios | Words | Content |
|---------|--------|------:|---------|
| W208 Surgical | ff. 1r–19r | ~7,500 | Surgical recipes (plasters, bandages, eye remedies) |
| W208 Antidotario | ff. 19r–51r | 13,651 | Compound drugs by therapeutic action |
| W208 Additional | ff. 51r–123v | 25,727 | Additional recipes (electuaries, unguents, waters, oils, pills) |
| **W208 Full** | **ff. 1r–123v** | **46,943** | **Complete manuscript** |

The Antidotario section is organised by therapeutic action (repercussive → resolutive → mundificative → glutinative → mollificative → cauteristic), the same organising principle as Circa Instans.

## 4. Method

### Test 1: Screening battery (m_core leading character distribution)

As S6, extended to Greek pharmaceutical text with and without function word filtering.

### Test 2: Forward cipher — clean baseline (S4)

The confirmed clean-baseline forward cipher (S4) was run with each source corpus using identical parameters: same grid (ROW_MAP), same VMS cell pools, same EC threshold (53%), same scoring suite (S5, 90 metrics with v2 tolerances). Five seeds (42, 404, 501, 606, 808) for each configuration.

### Test 3: Forward cipher — full v11 (S1)

The full v11 forward cipher including all scribe production rules (preferential reuse, suffix avoidance, boundary innovation, column stickiness) was run with CI Latin and all three W208 segments. Five seeds (42, 404, 501, 606, 808).

### Test 4: Component-level discrimination

Three tests designed to exploit structural differences that survive the cipher transform, tested across all six VMS sections:

**Suffix family distribution.** The confirmed mapping V₁→suffix family (a→Y, e→R, i→N, o→L, u→BARE) predicts suffix family proportions from source-language first-vowel distributions. Zero free parameters.

**C₁–C₁ bigram correlation.** Pearson correlation between predicted and observed bigram distributions of adjacent content-word initial consonants.

**Row distribution χ².** First-consonant group distribution of content words compared to VMS m_core leading character distribution per section.

### Test 5: Cell-sequence tests

Tests operating on the cell-assignment sequence (which cell each source word routes to), which the scribe layer cannot modify:

**EC/FC alternation.** EC rate, mean EC run length, mean FC run length.

**Row transition correlation.** FC→FC row adjacency patterns correlated with VMS.

**Family transition correlation.** Suffix family adjacency patterns correlated with VMS.

**Sandwiched FC row distribution.** Row distribution of FC tokens between two EC tokens.

**Conditional entropy.** H(row_i | row_{i-1}) for adjacent FC tokens. Verified by bootstrap (1000 line-level resamples), stability across 20 random starting offsets, and cross-section replication.

## 5. Results

### 5.1 Screening battery

| Corpus | N | χ² | Status |
|--------|--:|---:|--------|
| CI Latin (confirmed baseline) | 24,300 | 0.458 | S6 result |
| Greek pharma (all words) | 122,180 | 0.576 | Competitive |
| Greek pharma (no function words) | 81,649 | 0.138 | Lowest χ², but 3 free parameters |

The screening battery cannot reliably discriminate between Greek and Latin pharmaceutical text.

### 5.2 Forward cipher — clean baseline (S4)

**Greek vs Latin (→ Herbal-A):**

| Configuration | Mean score | C15 | Types |
|---------------|----------:|----:|------:|
| CI Latin + Latin nomenclator | 51.4/90 | 9.0 | 872 |
| Greek DMM (no nomenclator) | 52.2/90 | 9.6 | 800 |
| Greek DMM + Greek nomenclator | 52.2/90 | 10.4 | 807 |

**Wellcome 208 vs CI Latin (→ Herbal-A):**

| Configuration | Mean score | C15 | Types |
|---------------|----------:|----:|------:|
| W208 Antidotario | 57.7/90 | 12.7 | 850 |
| W208 Additional | 56.3/90 | 12.0 | 850 |
| CI Latin | 51.0/90 | 8.7 | 870 |

W208 Antidotario scores 6.7 points above CI Latin with non-overlapping ranges on the clean baseline.

**Cross-section (→ Stars):**

| Configuration | Mean score | C15 |
|---------------|----------:|----:|
| W208 Additional | 48.0/90 | 10.3 |
| CI Latin | 46.7/90 | 11.0 |
| W208 Antidotario | 46.0/90 | 9.7 |

### 5.3 Forward cipher — full v11 (S1)

| Configuration | Mean score | C15 | Types | Range |
|---------------|----------:|----:|------:|-------|
| W208 Full | 69.6/90 | 12.4 | 1432 | 66–74 |
| W208 Antidotario | 69.2/90 | 12.0 | 1432 | 66–72 |
| W208 Additional | 68.8/90 | 11.8 | 1433 | 65–71 |
| CI Latin | 68.4/90 | 11.8 | 1432 | 64–72 |

**The scribe production layer absorbs the source-language advantage.** Clean baseline: W208 +6.7 points over CI Latin. Full v11: +0.8 points with fully overlapping ranges. The scribe rules (copy-mutate, preferential reuse, suffix avoidance, column stickiness) dominate output statistics and normalise source-text differences.

**The v11 aggregate score validates the cipher architecture and scribe model, not the source language.** Source-language identification rests entirely on the component-level tests (§5.4) and cell-sequence tests (§5.5).

### 5.4 Component-level discrimination: Greek vs Latin

**Suffix family distribution (zero free parameters, Latin wins 6/6 sections):**

| Section | N_FC | Latin χ² | Greek χ² | Ratio |
|---------|-----:|--------:|---------:|------:|
| Herbal-A | 1,737 | 0.078 | 0.240 | 3.1× |
| Herbal-B | 2,538 | 0.189 | 0.390 | 2.1× |
| Pharmaceutical | 2,151 | 0.093 | 0.211 | 2.3× |
| Stars | 5,340 | 0.116 | 0.255 | 2.2× |
| Balneological | 2,501 | 0.523 | 0.672 | 1.3× |
| Zodiac | 1,006 | 0.326 | 0.350 | 1.1× |

The critical mismatch for Greek: R-family predicted at 31.1% vs VMS 16.7% (driven by η/ε → 'e'), and BARE predicted at 10.4% vs VMS 21.9% (rare υ). These are structural properties of the language.

**C₁–C₁ bigram correlation (Latin wins 5/6 sections):**

| Section | Latin r | Greek r | Δ |
|---------|--------:|--------:|---:|
| Herbal-A | 0.973 | 0.942 | +0.031 |
| Herbal-B | 0.881 | 0.817 | +0.064 |
| Pharmaceutical | 0.416 | 0.288 | +0.128 |
| Stars | 0.310 | 0.286 | +0.024 |
| Balneological | 0.059 | 0.075 | −0.016 |
| Zodiac | 0.174 | 0.074 | +0.101 |

**Row distribution χ² (Latin wins 5/6 sections):**

| Section | Latin χ² | Greek χ² | Ratio |
|---------|--------:|---------:|------:|
| Herbal-A | 0.021 | 0.078 | 3.7× |
| Herbal-B | 0.143 | 0.219 | 1.5× |
| Pharmaceutical | 0.677 | 1.021 | 1.5× |
| Stars | 1.138 | 1.187 | 1.0× |
| Balneological | 2.100 | 1.908 | 0.9× |
| Zodiac | 2.224 | 2.581 | 1.2× |

**Total: Latin wins 16 of 18 section × test comparisons. Greek wins 2 (Balneological only, margins within noise).**

### 5.5 Cell-sequence tests: CI Latin vs mixed Latin-Italian

**EC/FC alternation:**

| Source | EC% | EC run | FC run |
|--------|----:|-------:|-------:|
| VMS Herbal-A | 0.569 | 2.46 | 1.86 |
| W208 Antidotario | 0.541 | 2.12 | 1.80 |
| CI Latin | 0.476 | 1.76 | 1.93 |

W208 closer to VMS on EC rate and run length.

**Row and family transition correlations (CI wins):**

| Test | CI Latin | W208 Antidotario |
|------|--------:|----------------:|
| Row transition r | 0.938 | 0.898 |
| Family transition r | 0.923 | 0.852 |
| Sandwiched FC χ² | 0.078 | 0.276 |

**Conditional entropy H(row_i | row_{i-1}) — W208 wins, verified:**

VMS Herbal-A: H = 2.511 bits. Bootstrap 95% CI (1000 line-level resamples): [2.437, 2.565].

| Source | H | Status |
|--------|---:|--------|
| CI Latin | 2.348 | **Outside 95% CI** (too predictable) |
| W208 Antidotario | 2.510 | **Inside 95% CI** |

Stability: CI Latin mean 2.378 ± 0.029 across 20 random offsets; W208 mean 2.434 ± 0.040.

Cross-section (W208 wins 5/6):

| Section | VMS H | CI H | W208 H | \|VMS−CI\| | \|VMS−W208\| | Winner |
|---------|------:|-----:|-------:|--------:|-----------:|--------|
| Herbal-A | 2.511 | 2.348 | 2.510 | 0.163 | 0.001 | W208 |
| Herbal-B | 2.564 | 2.355 | 2.507 | 0.209 | 0.058 | W208 |
| Pharmaceutical | 2.447 | 2.347 | 2.514 | 0.100 | 0.067 | W208 |
| Stars | 2.649 | 2.365 | 2.486 | 0.284 | 0.163 | W208 |
| Balneological | 2.646 | 2.345 | 2.514 | 0.301 | 0.132 | W208 |
| Zodiac | 2.077 | 2.302 | 2.503 | 0.225 | 0.426 | CI |

### 5.6 Summary of discriminating tests

| Test | CI Latin | W208 | Greek | Measures |
|------|:--------:|:----:|:-----:|----------|
| Suffix family χ² | ✓ 6/6 | — | 0/6 | V₁ distribution |
| C₁–C₁ bigram r | ✓ 5/6 | — | 1/6 | Consonant adjacency |
| Row distribution χ² | ✓ 5/6 | — | 1/6 | Consonant group frequencies |
| Row transition r | ✓ | | — | FC→FC patterns |
| Family transition r | ✓ | | — | V₁→V₁ patterns |
| Sandwiched FC χ² | ✓ | | — | Content between FWs |
| **Conditional entropy** | | **✓ 5/6** | — | **C₁ sequence information density** |
| EC% | | ✓ | — | Function word rate |
| EC run length | | ✓ | — | Function word clustering |

CI Latin wins on *which patterns* appear. W208 wins on *how much information* they carry.

## 6. Discussion

The central finding is that the forward cipher v11 aggregate score cannot discriminate between source texts. The scribe production layer normalises output statistics so effectively that pharmaceutical Latin, Greek pharmaceutical text, and mixed Latin-Italian all score in the same band (68–70/90).

Discrimination requires tests that bypass the scribe layer. Two classes succeed:

**Component-level tests** (suffix family, bigram correlation, row distribution) separate Greek from Latin cleanly (Latin 16/18, Greek 2/18) and confirm pharmaceutical Latin.

**Cell-sequence tests** (conditional entropy, EC/FC alternation) reveal that pure pharmaceutical Latin is too structured. Its consonant transitions are more predictable than the VMS requires (H = 2.348 vs VMS 2.511, outside bootstrap 95% CI). Mixed Latin-Italian matches the VMS entropy (H = 2.510) in 5 of 6 sections.

The two sets of results are not contradictory. The VMS source text has the consonant adjacency *patterns* of Latin (CI wins on correlation) at the information *density* of Italian (W208 wins on entropy). This is consistent with Latin pharmaceutical terminology embedded in Italian syntactic structure — a macaronic register well attested in fifteenth-century Paduan recipe collections.

The nomenclator validation (r = 0.96, main paper §5.5) remains Latin-specific: the function word assignments (*et*, *in*, *cum*, *de*, *ad*, *habet*, *uel*, *que*, *supra*) are Latin regardless of surrounding syntax.

## 7. Conclusion

Greek is excluded by component-level tests (Latin 16/18 vs Greek 2/18). The v11 aggregate score validates architecture, not language; this limitation should be noted when interpreting v11 results. An information-theoretic test reveals pure pharmaceutical Latin is too structured for the VMS (p < 0.05, bootstrap); mixed Latin-Italian matches in 5/6 sections. The source text's register may be macaronic, consistent with the Padua–Pavia–Swabia/Bavaria production triangle.

## Data availability

| File | Description |
|------|-------------|
| `greek_battery_v2_2.py` | Battery and forward cipher code |
| `greek_corpus_parsed.pkl` | Romanised Greek pharmaceutical corpus (122k words) |
| `greek_dmm_corpus.pkl` | Romanised Dioscorides DMM (112k words) |
| `dmm_definitive_results.pkl` | Forward cipher scoring: Greek vs Latin |
| `w208_multivalent_results.pkl` | Forward cipher scoring: W208 segments × VMS sections |
| `w208_transkribus_export.zip` | Wellcome 208 Transkribus PAGE XML |

Greek texts from OpenGreekAndLatin/First1KGreek (MIT licence) and the Wellmann/Kühn edition (public domain). Wellcome 208 from Wellcome Collection digital images (Public Domain Mark).
