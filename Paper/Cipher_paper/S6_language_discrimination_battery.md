# Supplement S6: Language Discrimination Battery

## Method

The VMS Herbal-A FC (content word) tokens were decomposed using PGCS. The `m_core`
leading character for each FC token gives the consonant row in the babuini grid.
The distribution across 7 rows (o, c, e, a, d, l, r) was compared against 16
candidate source languages using χ² goodness-of-fit.

Each candidate text was tokenised, consonant-classified using the same grid mapping,
and its row distribution compared against the VMS target.

## VMS Target Distribution (Herbal-A FC tokens)

| Row | VMS %  | Consonants in row |
|-----|--------|-------------------|
| o   | 35.4%  | c, s, p           |
| c   | 28.9%  | ∅, v              |
| e   | 15.2%  | f, d              |
| a   |  9.3%  | m, l              |
| d   |  5.9%  | r, q, h, n, g     |
| l   |  3.3%  | t                 |
| r   |  1.5%  | b, z, x, j, k, w, y |

## Results (ranked by χ²)

| Rank | Language | Corpus | N tokens | χ² |
|------|----------|--------|----------|----|
| 1 | Latin (pharma) | Circa Instans | 24,300 | 0.0249 |
| 2 | Italian vernac. | Dante Commedia | 89,823 | 0.0868 |
| 3 | Occitan | Leys d'Amors | 120,012 | 0.1323 |
| 4 | Portuguese | Os Lusíadas | 49,491 | 0.1732 |
| 5 | Spanish | Don Quijote | 354,176 | 0.1900 |
| 6 | Greek (Diosc.) | Dioscorides preface | 465 | 0.2051 |
| 7 | Latin (biblical) | Vulgate Gen+Exod | 45,557 | 0.2173 |
| 8 | Latin (rhetoric) | Cicero In Catilinam | 12,490 | 0.2478 |
| 9 | MHG | Nibelungenlied | 250,915 | 0.5527 |
| 10 | Middle English | Canterbury Tales | 269,139 | 0.6412 |
| 11 | English (Early Mod) | Bacon Novum Organum | 92,831 | 0.6914 |
| 12 | Catalan | Tirant lo Blanch | 157,659 | 0.8290 |
| 13 | Old French | original sample | 5,174 | 0.9970 |
| 14 | Hebrew | Guide of the Perplexed | 168,391 | 1.2315 |
| 15 | Arabic | Guide (Arabic) | 168,391 | 1.3336 |
| 16 | Arabic | Ibn al-Arabi | 2,731,129 | 1.7399 |

## Interpretation

Pharmaceutical Latin (CI) is the best fit by a factor of 3.5× over the next candidate
(Italian vernacular). Within Latin, the pharmaceutical register (CI) fits 8.7× better
than biblical Latin and 10× better than rhetorical Latin, indicating genre specificity.

Germanic languages (MHG, ME, Early Modern English) cluster at χ² > 0.55, consistent
with the independent sonorant concentration test (§8 main text) which eliminates
Germanic at 44-50% vs VMS 93.3%.

Semitic languages (Hebrew, Arabic) show the worst fit (χ² > 1.2), as expected from
their radically different consonant distributions.

## Note on χ² values

The χ² values here use the proportional method: Σ(obs-exp)²/exp where obs and exp
are row proportions. The paper's §8 reports a slightly different value (0.039) computed
with sample-size weighting on the full 1,621 FC token sample. Both methods rank CI first.

## Data

Full distributions for all 16 languages are in S6_language_battery_results.json.
