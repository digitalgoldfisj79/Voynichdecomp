# VBM v146 · compact scribal abbreviation and positional encoding screen
14 October 2026 — **engineering exploration; surface NO-GO; historical identification NOT QUALIFIED**

## Summary and corrections

- **The first four-shortcut A improvement was not stable:** one 950-swap annealing seed found 39.15% illegal-trigram words versus 41.18% for the corresponding 23-symbol baseline. A separate development-validation-selected, equal-budget two-seed comparison instead found **45.31%** versus **35.81%**. **Do not promote the first seed.** This optimisation is stochastic, low-budget and not certified globally optimal.
- With the common development-only exploratory grid, **none of the two/four/six shortcut codebooks beat the uncontracted 23-sign comparator on the complete surface/length objective** in either Currier class. One escape sign is required when 23 source letters and added digraph tokens must be represented with only 23 surface atoms by a single, self-delimiting lookup table.
- Nearly every word containing an escape code has an unsupported native trigram (using the tested `g + glyph` marker). This identifies a construction defect, not a theorem that scribal abbreviation is impossible.
- A 69-entry, **three-position-column** codebook (one 23-letter permutation for the first, interior and final source-character positions) is also exactly decodable without escapes, but leaves **33.38% A and 32.42% B** words trigram-illegal, with output averages 5.442 A / 5.426 B and tails still much too long. More tables did not solve the problem.
- All target statistics are against Voynich reference inventories **already used for fitting**. The German Hartlieb text is repeatedly inspected by the v142–v146 programme and not a fresh prospective holdout. No inference about cipher likelihood follows. No ductus, pixel joins, real-hand nor true scribe-speed qualification has been performed.

## Input specification

Original v142–v146 German source: 68,396 development words and 13,232 Hartlieb words, normalized to 23 lower-case letters, all source files in `inputs/`. The native A and B comparison inventories are the original frozen **REVA** files, 10,995 A words and 21,678 B words, using the same glyph-atom reversal convention (`ch,sh,ckh,cth,cph,cfh` greedy). These are pooled language-A/B references, not section-matched held-out folios. Frozen input hashes in `INPUT_HASHES.json`.

All source token substitutions are injective, all decoding uses only supplied codebook and ciphertext. The preselected ordinary digraph contractions are `en`, `er`, `ch`, `un`, followed in the exploratory grid by the prefix `en,er`, six `en,er,ch,un,nd,ge` and a matched-size alternative `de,an,te,st`.

A 23-sign uncontracted model uses one unique glyph atom for each source character; an abbreviated model uses the 22 other atoms for 22 frequent source tokens, and maps the remaining `K+1` source tokens to distinct `escape atom + following atom` pairs, with one escape atom reserved. This gives 23+K distinguishable symbol codes using 23 glyph atoms. Word segmentation remains one source word → one output word. The greedy abbreviation parser is unambiguous as long as the decoder reverses the token table. All models are exact (13,232/13,232 Hartlieb words), not manuscript-deciphering instruments.

The optimiser makes letter/sign swaps with a fixed annealing schedule, assessing on 1/40 of German development words, choosing the seed via a disjoint 1/37 development sample, with Hartlieb measured only diagnostically. Objective includes unsupported trigram-word rate, bigram-word rate, invalid boundary forms, invalid trigram occurrences, squared mean-length discrepancy and >9-character tail excess. Each variant uses the same two seeds and 700-swap optimisation budget. **This is a limited stochastic search**, so it cannot certify the global optimum or a mathematical nonexistence result.

## Common-grid Hartlieb diagnostics

| Class | Scheme | Table entries | Mean raw EVA chars / word | Words with invalid trigram | Words ≥10 chars |
|---|---|---:|---:|---:|---:|
| A | 23-symbol baseline | 23 | 5.295 | 35.81% | 9.60% |
| A | `en,er` | 25 | 5.631 | 38.66% | 13.70% |
| A | `en,er,ch,un` | 27 | 5.437 | 45.31% | 13.00% |
| A | `en,er,ch,un,nd,ge` | 29 | 5.269 | 49.98% | 13.80% |
| A | alternate four | 27 | 5.761 | 44.20% | 16.00% |
| B | 23-symbol baseline | 23 | 5.711 | 31.77% | 12.40% |
| B | `en,er` | 25 | 5.562 | 41.66% | 12.20% |
| B | `en,er,ch,un` | 27 | 5.353 | 37.52% | 13.00% |
| B | `en,er,ch,un,nd,ge` | 29 | 5.732 | 41.91% | 15.10% |
| B | alternate four | 27 | 5.520 | 41.91% | 14.20% |

Native A reference mean = 4.783, ≥10 chars ≈0.56%; native B mean = 5.061, ≥10 chars ≈0.81%. The grid table is the comparable exploratory run; the original one-seed 950-swap run showed more favourable, but unstable A and B optima (A simple 41.18 vs four 39.15; B simple 30.23 vs four 38.85). This difference in stochastic optimisation seeds means neither table estimates a model-class minimum.

## Why fixed abbreviations did not automatically help

The source-unit count decreases (Hartlieb: 4.079 letter units per word at baseline, 3.600 units with four digraph shortcuts). But reserving a glyph for an escape means rare letters or other source tokens now require *two* output glyph atoms. In the 4-shortcut model 1,940/13,232 words contain an escape. Almost all these words contain an invalid trigram. On the equal-grid A model, 1,939 of those 1,940 escape-bearing words are invalid; on B, all 1,940 are. Thus the shorthand gain is largely consumed by artificial long codes and inadmissible joins.

Varying the escape sign from `g` to `l` or `o` in the frozen four-shortcut B construction did not resolve the admission problem: on Hartlieb, bad-trigram shares were 41.04%, 37.36%, 39.58% respectively and mean lengths 5.577, 5.711, 5.523. Selection of an escape based on target inventory is yet another fit, not a historical key.

## Positional scribal alphabet diagnostic

A three-column table uses one source-letter-to-glyph mapping for first, interior, or last *encoded* glyph positions. Total = 69 substitution entries, exactly reversible with independent inverse column tables. On Hartlieb:

| Class | Invalid trigram-word share | Invalid bigram-word share | Mean characters | ≥10 chars | Entries |
|---|---:|---:|---:|---:|---:|
| A | 33.38% | 10.17% | 5.442 | 11.8% | 69 |
| B | 32.42% | 7.66% | 5.426 | 10.0% | 69 |

This can reduce illegal *bigrams*, but does not qualify the full-word, length or positional grammar. Furthermore, `REVA` reverses the order of EVA compound atoms. A scribe actually working left-to-right in the physical script would have to reverse source-word reading order or use an equivalent mirror-transducer to realise this model. The experiment **does not demonstrate an executable production workflow in actual handwriting**.

## Mechanism and identifiability verdict

**Finding:** Shortening source units is in principle compatible with exact decoding and lower codebook complexity, but a few uncontextualised digraph substitutions do not repair Voynich word grammar. The escape marker is the main identifiable bottleneck of the particular trial. Neither simple substitution nor three positional substitution columns achieves whole-word surface fidelity. The prior v145 architecture remains retired under its frozen segmentation; v143/v144's cross-word engineering success remains valid but is not restored as a manuscript mechanism by these tests.

**Next mathematically discriminating task, if pursued:** a fixed compact, prefix/suffix-aware finite-state code with a *self-delimiting, observed-legal* abbreviation strategy must beat the 23-sign baseline on full word support, long-word tails, all sections, and independent physical writing joins. Before constructing it, bound how many distinct source units can be assigned to legal short forms at a given state and word position. A model of 23 source characters plus four freely chosen digraphs requires at least 27 different distinguishable values somewhere; it cannot fit into 23 unconditioned single-atom codewords by mere assertion. An alternative could exploit an externally evidenced positional state or a rigorously specified reversible shorthand, but cannot hide a free side channel or hundreds of exceptions.

**Decision:** There is **no qualified v146 scribal solution yet**. The current data should not be promoted as a cipher nor as a successful manuscript-fit model. Another blind VMS test is not justified until the prospective instrument beats a hard matched baseline and demonstrates a physically credible writing procedure.

## Reproduction

All fixed inputs, hashes, exact codebooks, numerical JSON, and prototype Python scripts are bundled. The canonical experimental outputs came from the archived working directory. The bundled standalone `scripts/position_model.py` was rerun in its extracted location for A and produced exactly the archived headline statistics. Code contains deterministic random seeds; there are no external API requirements.

Suggested commands from bundle root:

```bash
python scripts/test_abbreviations.py
python scripts/extended.py A zero
python scripts/extended.py B four
python scripts/escape_search.py B l
python scripts/position_model.py A
python scripts/position_model.py B
```

Not all of these commands were replayed *from the bundled copy*; exact numeric full bundle regeneration has not been independently confirmed beyond the above A positional recheck. Do not label it fully reproduced without running all.