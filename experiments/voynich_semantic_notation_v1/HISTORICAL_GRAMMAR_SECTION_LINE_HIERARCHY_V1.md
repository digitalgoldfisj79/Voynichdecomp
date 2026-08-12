# VSN-B2-v1 — Section → Line Hierarchy Test

Date: 2026-08-12
Status: COMPLETE FIRST HIERARCHICAL PASS
Primary historical mechanism: frozen Matteo da Verona K=2 first-syllable concatenation.

## 0. Why this test was added

The whole-corpus structural comparison could conceal strong Voynich section heterogeneity. The hierarchy was therefore rerun without modifying the historical generator:

1. section-by-section, matching each Voynich section's unique type count;
2. line-by-line within section;
3. running-text (`layout_family='P'`) separated from diagram/list/label loci;
4. hostile token shuffles within section × running-text layout preserving the actual token multiset and every line length.

No historical parameter was fit by section or line.

## 1. Edit-pair definition correction

The existing `rf_edit1_pairs` table contains 28,435 **edit-path rows** but 27,307 **distinct unordered token pairs**. There are 1,128 duplicate paths, mainly where the same pair admits more than one edit alignment.

The historical simulator stores distinct unordered pairs. Therefore all hierarchy analyses use 27,307-style distinct pairs, with duplicate Voynich paths collapsed. Where duplicate paths disagree on position class, the pair is classified `internal`, mirroring the simulator's conservative ambiguity handling.

This supersedes the earlier use of 28,435 as though it were a distinct-pair count. The whole-corpus density comparison actually becomes numerically closer (27,307 Voynich distinct pairs vs 26,094 Matteo K2), but the hierarchy test below shows that this aggregate closeness is misleading.

## 2. Section-by-section Voynich morphology

Corrected distinct-pair results:

| section | types | Voynich pairs | prefix | internal | suffix |
|---|---:|---:|---:|---:|---:|
| Stars | 3,121 | 8,794 | 0.3226 | 0.4867 | 0.1907 |
| Herbal-A | 2,812 | 7,832 | 0.3183 | 0.4671 | 0.2146 |
| unclassified/missing | 2,494 | 6,099 | 0.2833 | 0.5296 | 0.1871 |
| Balneological | 1,406 | 3,702 | 0.4133 | 0.4481 | 0.1386 |
| text-only | 907 | 1,745 | 0.3696 | 0.4424 | 0.1880 |
| Pharmaceutical | 566 | 786 | 0.2723 | 0.4517 | 0.2761 |
| Herbal-B | 437 | 563 | 0.3286 | 0.4156 | 0.2558 |
| Cosmological | 249 | 391 | 0.5575 | 0.2506 | 0.1918 |
| Zodiac | 223 | 169 | 0.3314 | 0.4675 | 0.2012 |

Section positional/transition structure also varies strongly. `H(next|prev)` ranges from 2.0507 bits (Zodiac) to 2.4370 (unclassified); `H(right position)-H(left position)` ranges from -0.2339 bits (Herbal-B) to +0.0168 (Stars). Literal Matteo K2 remains much less locally constrained (`H(next|prev)=3.3818`) and has the opposite global right-position sign (+0.0742).

## 3. Matched-size Matteo K2 section nulls

For each section, the frozen K2 generator was run for 20 deterministic seeds at exactly that section's Voynich type count. Same Latin lexicon, syllabifier, K=2 rule and sampling regime as `MATTEO_SIMULATION_SPEC_V1.md`; only output inventory size changes.

| section | Voynich pairs | Matteo mean | Matteo max (20) | V/M ratio | pair-count z | edit-location TV |
|---|---:|---:|---:|---:|---:|---:|
| Stars | 8,794 | 5,409.7 | 5,862 | 1.63 | 20.4 | 0.014 |
| Herbal-A | 7,832 | 4,476.9 | 4,631 | 1.75 | 38.5 | 0.028 |
| unclassified/missing | 6,099 | 3,685.0 | 3,876 | 1.66 | 18.8 | 0.032 |
| Balneological | 3,702 | 1,288.1 | 1,410 | 2.87 | 36.7 | 0.091 |
| text-only | 1,745 | 562.4 | 657 | 3.10 | 26.9 | 0.054 |
| Pharmaceutical | 786 | 226.9 | 294 | 3.46 | 19.8 | 0.080 |
| Herbal-B | 563 | 140.8 | 196 | 4.00 | 18.5 | 0.062 |
| Cosmological | 391 | 48.9 | 83 | 8.00 | 30.7 | 0.237 |
| Zodiac | 169 | 38.7 | 61 | 4.37 | 15.1 | 0.012 |

### Section verdict

Every Voynich section is substantially denser in exact one-edit neighbours than a same-size Matteo K2 inventory. Every observed section exceeds the maximum of all 20 frozen K2 replicates.

The edit-*location* geometry nevertheless remains close to K2 in several sections, especially Stars and Zodiac, and moderately close in Herbal-A/unclassified. It diverges strongly in specialised regimes, especially Cosmological and Balneological.

**Therefore the earlier whole-corpus pair-density match is an aggregation effect.** Pooling heterogeneous section vocabularies dilutes section-local morphological clustering until the union happens to approach K2 density. The whole-corpus 26,094-vs-27,307 similarity must not be interpreted as evidence that literal K2 generates Voynich's productive morphology.

The more defensible transfer is narrower: K2 reproduces a surprisingly similar *global edit-location geometry*, not the hierarchical density process.

## 4. Line-by-line analysis

A corrected line table was materialised in Supabase:

`voynich_semantic_notation_v1.line_edit_metrics_v1`

One row per RF locus/line, containing:

- section and layout family;
- folio/locus/line number;
- token and unique-type counts;
- adjacent-word opportunities/hits;
- all within-line pair opportunities/hits;
- section×layout weighted random-edit baseline;
- adjacent and arbitrary-pair enrichment.

All line calculations use distinct unordered edit pairs.

## 5. Section-conditioned local clustering

Using all loci, within-line edit-neighbour pairs are enriched relative to two tokens drawn from the same section's actual frequency distribution:

| section | within-line pair rate | section baseline | enrichment |
|---|---:|---:|---:|
| Herbal-A | 0.03231 | 0.01703 | 1.90× |
| Stars | 0.02659 | 0.01739 | 1.53× |
| unclassified | 0.02396 | 0.01390 | 1.72× |
| Balneological | 0.03897 | 0.02949 | 1.32× |
| text-only | 0.02922 | 0.01645 | 1.78× |
| Pharmaceutical | 0.03323 | 0.01670 | 1.99× |
| Herbal-B | 0.02054 | 0.01118 | 1.84× |
| Cosmological | 0.35425 | 0.08119 | 4.36× |
| Zodiac | 0.01701 | 0.01257 | 1.35× |

Adjacent-word enrichment is also positive in every section at this level.

## 6. Diagram/list outlier control

Cosmological is dominated by diagrammatic/circular loci rather than ordinary running text.

- f57v `<f57v.3,+Cc>` alone contributes 1,339 of 1,852 Cosmological within-line edit hits = 72.3%.
- the top five Cosmological loci contribute 98.6%.

By contrast the top individual line contributes only ~0.7–1.1% of hits in Stars, Herbal-A and Balneological.

After splitting by layout family, ordinary running text (`P`) gives:

| section | running-text pair enrichment |
|---|---:|
| Herbal-A | 1.92× |
| Herbal-B | 1.84× |
| Stars | 1.53× |
| text-only | 1.81× |
| Pharmaceutical | 1.60× |
| Balneological | 1.30× |
| Cosmological | 1.19× (only 35 tokens) |

Thus the extreme Cosmological effect is not a general line-text property.

## 7. Running-text local edit positions

Among actual within-line one-edit hits in `P` loci:

| section | prefix | internal | suffix |
|---|---:|---:|---:|
| Stars | 0.3850 | 0.4742 | 0.1409 |
| Balneological | 0.4409 | 0.4307 | 0.1284 |
| Herbal-A | 0.4159 | 0.3401 | 0.2440 |
| unclassified | 0.4467 | 0.3099 | 0.2435 |
| text-only | 0.3650 | 0.4200 | 0.2150 |
| Pharmaceutical | 0.3308 | 0.3158 | 0.3534 |
| Herbal-B | 0.2581 | 0.3548 | 0.3871 |
| Cosmological | 0.4000 | 0.4000 | 0.2000 | 

Matteo K2's whole-inventory profile is approximately 0.305/0.510/0.186. The line-local Voynich distributions therefore diverge much more strongly and in section-specific directions. Herbal-B and Pharmaceutical are locally suffix-heavy; Balneological and Herbal-A are locally prefix-heavy.

This is evidence that the same global edit topology is produced by multiple local regimes rather than one homogeneous K2-like process.

## 8. Hostile within-section line shuffle

A 64-permutation deterministic null shuffled the **actual Voynich tokens** among running-text lines within each section while preserving:

- the section token multiset and all token frequencies;
- layout family = P;
- every line length;
- total token count.

Observed vs null within-line edit hits:

| section | observed | null mean | null p99 | >= observed /64 | empirical p |
|---|---:|---:|---:|---:|---:|
| Stars | 1,278 | 836.2 | 907.3 | 0 | 1/65 = 0.01538 |
| Balneological | 880 | 674.5 | 712.0 | 0 | 0.01538 |
| Herbal-A | 832 | 429.8 | 471.4 | 0 | 0.01538 |
| unclassified | 497 | 293.6 | 329.7 | 0 | 0.01538 |
| text-only | 200 | 110.1 | 134.5 | 0 | 0.01538 |
| Pharmaceutical | 133 | 83.4 | 103.2 | 0 | 0.01538 |
| Herbal-B | 31 | 17.0 | 24.4 | 0 | 0.01538 |
| Cosmological | 5 | 4.09 | 8.37 | 27 | 0.43077 |

A 256-permutation combined query was attempted after this and hit the Supabase statement timeout. It was read-only and changed no data. No threshold or model was altered; the completed 64-permutation null is the reported result.

## 9. Revised scientific interpretation

The hierarchy test materially revises the earlier K2 result.

### What survives

1. Matteo K2 is an independently attested Padua-1420 compositional artificial-word mechanism.
2. It naturally produces a dense one-edit graph unlike iid characters or ordinary Latin vocabulary.
3. Its global prefix/internal/suffix edit geometry is unexpectedly close to whole-corpus Voynich and remains close in some individual sections.

### What does not survive

1. K2 does **not** reproduce the section-local edit density: every section is much denser.
2. K2 does **not** reproduce the section-specific local regimes visible inside lines.
3. Voynich morphological neighbours are actively concentrated within ordinary running-text lines even after conditioning on section frequencies; a word-formation rule with no line assembly mechanism does not explain this.
4. Voynich's stronger Markov/positional constraints remain unexplained.

### Verdict

**Matteo K2 remains a genuine historical-mechanism / partial-topology transfer, but the hypothesis that literal K2 is the Voynich generative mechanism is now substantially weakened.**

The most interesting positive finding has shifted from "Voynich has K2-like density" to:

> A historically attested two-component syllabic composition rule generates roughly the same global edit-location topology as Voynich, while Voynich superimposes strong section- and line-specific morphological selection on top of that topology.

That is a sharper target for further historical work. Any serious analogue now needs not only component composition, but **context-dependent selection of components by technical domain and local record/line context**.
