# VSN-B2-v1 Historical Grammar Structural Comparison — Results

Date: 2026-08-12
Status: CLOSED FIRST STRUCTURAL PASS
Protocol: `HISTORICAL_GRAMMAR_PROTOCOL_V1.md`
Simulation preregistration: `MATTEO_SIMULATION_SPEC_V1.md`

## Executive result

The source-faithful historical comparison yields one genuine **PARTIAL TRANSFER** and three **MECHANISM-ONLY** outcomes.

The nontrivial result is Matteo da Verona's attested artificial-word operation. When instantiated literally as concatenation of **two first syllables** drawn from an independent Latin vocabulary, it reproduces the Voynich RF exact-edit neighbourhood geometry unusually closely without parameter fitting:

- exact edit-1 pairs: 26,094 synthetic vs 28,435 Voynich;
- mean edit-1 degree: 6.612 vs 7.205;
- isolated-type fraction: 0.1263 vs 0.1491;
- edit-location distribution:
  - prefix 30.48% vs 29.07%;
  - internal 50.95% vs 52.78%;
  - suffix 18.57% vs 18.15%.

The total-variation distance between the three-way edit-location distributions is only **0.0183**.

This is not reproduced by a source-character iid control, nor is it simply the geometry of the underlying Latin vocabulary. But literal K=2 Matteo does **not** reproduce the full Voynich surface system: it is too short/narrow at type level, has much weaker character-transition constraints, and lacks Voynich's global right-edge positional constraint. K=3, K=5 and the equal 2/3/5 mixture fail much more strongly.

Therefore the result is **PARTIAL TRANSFER, not STRUCTURAL TRANSFER**.

No separate historical systems are combined to repair the mismatch.

---

## 1. Frozen Voynich target panel

Primary target: `voynich_semantic_notation_v1.rf_token_types`, exact-letter RF types only.

### 1.1 Type inventory / length

- types: 7,893;
- mean type length: 6.4968;
- p10 / median / p90: 4 / 6 / 9 characters;
- minimum / maximum: 1 / 20.

For context only, occurrence-weighted mean token length is 5.1259; historical simulations do not possess attested token-frequency weights, so type-weighted comparison is primary.

### 1.2 Exact edit-1 graph

- edit-1 pairs: 28,435;
- mean degree: 7.2051;
- median degree: 5;
- isolated fraction: 0.149119;
- maximum degree: 54.

Edit-location distribution:

- prefix: 29.0662%;
- internal: 52.7836%;
- suffix: 18.1502%.

### 1.3 Character / positional structure — type weighted

- alphabet size: 24;
- H(character): 3.89535 bits;
- H(character | absolute position from left): 3.57340;
- H(character | position from right): 3.42075;
- H(first character): 3.39729;
- H(last character): 2.83100;
- H(next character | previous character): 2.53552;
- observed bigram types: 282.

The right-position conditional entropy is 0.15266 bits **lower** than the left-position conditional entropy: Voynich has a real global right-edge positional constraint in this representation.

---

## 2. Matteo simulation source and compute

Independent source vocabulary:

- `sjgallagher2/PyWORDS/pywords/data/lingualatina_voclist.txt`;
- Git blob recorded at preregistration: `5dc8e924f253ef18cc72d72daa15ec49a805b8f8`;
- fetched raw-file SHA-256 during execution: `5a139a6e7a3b9bfe9ef0b0e98e5178fb1c42be66dc3034c3f6f5e3d91b099b9c`;
- 1,902 source lines;
- 1,846 eligible unique normalized source words;
- 429 unique derived first syllables.

The frozen orthographic syllabifier reproduces the source examples:

- `tripode -> tri`;
- `pepo -> pe`;
- `corvus -> cor`;
- `vetula -> ve`.

No Latin item or syllable was selected because of Voynich resemblance.

Primary compute was a bounded Hugging Face CPU job, job id `6a7bbf9527caad61c6eaca79`; it completed normally. A 20-seed robustness CPU job `6a7bbfc8f6d0f3ee953aa36a` also completed normally. A source-Latin control job `6a7bbfedf6d0f3ee953aa372` completed normally. Final job-status check: **no running jobs**.

---

## 3. Primary Matteo results

All runs contain 7,893 unique generated types.

| sampling | slots | mean len | p10/med/p90 | edit pairs | mean deg | isolated | prefix | internal | suffix | H(next|prev) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| lemma-uniform | **2** | **4.913** | 4/5/6 | **26,094** | **6.612** | **0.1263** | **0.3048** | **0.5095** | **0.1857** | 3.382 |
| lemma-uniform | 3 | 7.106 | 5/7/9 | 622 | 0.158 | 0.8852 | 0.2010 | 0.6736 | 0.1254 | 3.462 |
| lemma-uniform | 5 | 11.814 | 10/12/14 | 0 | 0 | 1.0000 | — | — | — | 3.535 |
| lemma-uniform | equal 2/3/5 | 8.040 | 4/7/13 | 4,134 | 1.048 | 0.7348 | 0.3060 | 0.5075 | 0.1865 | 3.488 |
| unique-syllable-uniform | 2 | 5.803 | 5/6/7 | 4,431 | 1.123 | 0.3746 | 0.2492 | 0.5653 | 0.1855 | 3.455 |
| unique-syllable-uniform | 3 | 8.707 | 7/9/10 | 15 | 0.0038 | 0.9962 | 0 | 0.9333 | 0.0667 | 3.560 |
| unique-syllable-uniform | 5 | 14.510 | 13/15/16 | 0 | 0 | 1.0000 | — | — | — | 3.623 |
| unique-syllable-uniform | equal 2/3/5 | 9.699 | 5/9/15 | 515 | 0.130 | 0.8954 | 0.2155 | 0.5903 | 0.1942 | 3.576 |

### 3.1 K=2 is qualitatively different

Only the lemma-uniform K=2 instantiation produces a dense edit graph in the Voynich range. K=3 becomes mostly isolated and K=5 entirely isolated at this sample size. The equal mixture inherits too much of the longer-component sparsity.

This is not a tuning decision: K=2, K=3, K=5 and equal mixture were frozen and all are reported.

### 3.2 Edit geometry — the strongest transfer

K=2 vs Voynich:

| metric | Voynich | Matteo K2 | difference |
|---|---:|---:|---:|
| edit pairs | 28,435 | 26,094 | -8.23% |
| mean degree | 7.205 | 6.612 | -8.23% |
| isolated fraction | 0.1491 | 0.1263 | -0.0228 |
| prefix edits | 0.2907 | 0.3048 | +0.0141 |
| internal edits | 0.5278 | 0.5095 | -0.0183 |
| suffix edits | 0.1815 | 0.1857 | +0.0042 |

Three-way edit-location total variation = **0.0183**.

The close location profile is not simply inherited unchanged from the raw Latin source vocabulary. The eligible source Latin word list itself has:

- prefix 38.03%;
- internal 41.62%;
- suffix 20.35%;

with edit-location TV distance from Voynich ≈ **0.1116**. Two-first-syllable composition moves that geometry substantially toward the Voynich distribution.

### 3.3 Robustness across 20 additional seeds

A separate 20-seed diagnostic, with no seed selection, gives:

- edit pairs mean 25,637.3, SD 366.7, range 25,067–26,728;
- mean degree 6.4962, SD 0.0929;
- isolated fraction 0.13068, SD 0.00475;
- prefix fraction 0.30717, SD 0.00355;
- internal fraction 0.50871, SD 0.00330;
- suffix fraction 0.18412, SD 0.00271;
- mean length 4.9209, SD 0.0077.

Thus the edit-location result is not a lucky deterministic sample.

---

## 4. Hostile controls

### 4.1 iid characters

At the K=2 mean length:

| control | edit pairs | mean degree | isolated | prefix | internal | suffix | H(next|prev) |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid uniform Latin alphabet | 485 | 0.123 | 0.8846 | 0.1835 | 0.6227 | 0.1938 | 4.573 |
| iid source-character marginal | 4,878 | 1.236 | 0.4263 | 0.2001 | 0.5951 | 0.2048 | 3.962 |
| Matteo K2 | 26,094 | 6.612 | 0.1263 | 0.3048 | 0.5095 | 0.1857 | 3.382 |
| Voynich | 28,435 | 7.205 | 0.1491 | 0.2907 | 0.5278 | 0.1815 | 2.536 |

The dense edit neighbourhood and its location profile are therefore not generated by iid characters with the same size/mean length.

### 4.2 Ordinary Latin vocabulary

The 1,846 eligible source Latin types themselves have:

- 752 edit-1 pairs;
- mean degree 0.815;
- isolated fraction 0.635;
- prefix/internal/suffix = 0.3803/0.4162/0.2035;
- mean length 6.686.

Different inventory size limits direct density comparison, but the edit-location profile is plainly not the K=2/Voynich profile. The compositional transformation itself changes the topology.

---

## 5. Where literal Matteo K=2 fails

The edit-graph match is not a full surface match.

### 5.1 Type-length distribution

Voynich RF types:

- mean 6.497;
- p10/median/p90 = 4/6/9.

Matteo K2:

- mean 4.913;
- p10/median/p90 = 4/5/6.

K2 is substantially shorter and far narrower in its upper tail. The occurrence-weighted Voynich mean (5.126) is close to K2, but token-frequency weights are unavailable for the historical generator, so that is secondary context, not the primary comparison.

### 5.2 Local transition constraint

Type-weighted first-order conditional entropy:

- Voynich `H(next|prev)` = 2.5355 bits, conditional perplexity ≈ 5.80;
- Matteo K2 = 3.3818 bits, perplexity ≈ 10.42.

Voynich has much stronger local character dependence.

Observed bigram types:

- Voynich 282;
- Matteo K2 396.

### 5.3 Positional/right-edge structure

Voynich:

- H(char | left absolute position) = 3.5734;
- H(char | right position) = 3.4207;
- right-minus-left = **-0.1527 bits**.

Matteo K2:

- H(char | left position) = 3.8971;
- H(char | right position) = 3.9713;
- right-minus-left = **+0.0742 bits**.

Thus the global positional asymmetry goes in the wrong direction.

At the single boundary character level both systems do have lower final than initial entropy:

- Voynich first/last = 3.3973 / 2.8310;
- Matteo K2 first/last = 4.0962 / 3.7073.

So Latin syllable endings create some terminal restriction, but they do not generate the deeper right-edge conditional architecture observed in Voynich.

### 5.4 Overall character constraint

- Voynich H(character) = 3.8954 bits;
- Matteo K2 = 4.0641 bits.

Again, literal K2 is less constrained.

---

## 6. Historical-system verdicts

### Matteo da Verona — artificial composite words

**PARTIAL TRANSFER** for the K=2 literal surface operation.

Positive:

- independently attested in Padua c.1420/23;
- literal first-syllable composition creates a dense minimal-pair topology near Voynich;
- edit-location proportions are unexpectedly close and robust;
- effect strongly exceeds iid controls;
- raw Latin source vocabulary does not itself have the same edit-location profile.

Negative:

- K=3/K=5/mixed variants fail surface density/length;
- K=2 is too short/narrow at type level;
- local bigram constraint much too weak;
- global right-edge positional constraint absent/reversed.

This is evidence that a historically attested **artificial compositional token operation can naturally reproduce one major class of Voynich morphology**, not evidence that Voynichese is Matteo's system.

### Bartolomeo da Mantova — prepared four-syllable codewords

**MECHANISM ONLY / direct-surface MISMATCH pending exact inventory**.

The source describes a prepared table/codebook architecture, not free slot recombination. A 100-codeword prepared inventory cannot on its own explain 7,893 productive RF types and their dense family graph. Free recombination is prohibited because the source does not license it. Exact codeword transcription remains useful for a historical control, but cannot be used to promote the system by inventing productivity.

The secondary-source `400 words -> twenty codewords` statement remains rejected as internally inconsistent with the described 100-table architecture until manuscript/edition verification.

### Jacopo Ragona — typed spatial practical records

**MECHANISM ONLY**.

The debt grammar is a strong historical typed-field precedent, but the extracted source does not serialize its ten fields into linear written tokens. No linear grammar is invented.

### Vat. lat. 10488 — operational mathematical notation

**MECHANISM ONLY** for Voynich word morphology.

It establishes compact role/position-sensitive written technical notation in the 1424 Veneto, but it is an expression language rather than a word-token generator.

---

## 7. Revised scientific position

The B2 test narrows the structured-notation hypothesis substantially.

A historically exact Paduan artificial-token mechanism is now known that, under a literal and independently frozen instantiation, spontaneously generates **Voynich-like edit-neighbour topology and edit-position proportions**. That specific transfer is not reproduced by iid strings and is not merely the raw Latin vocabulary profile.

But the same mechanism does not produce Voynich's stronger sequence constraints, right-edge positional architecture, or long type tail. Therefore it cannot be treated as an adequate generative account.

The defensible inference is:

> The class of early-fifteenth-century northern-Italian artificial compositional token systems is no longer merely historically plausible; one attested member reproduces a nontrivial part of Voynich surface morphology without tuning. Additional structure would be required to reproduce Voynich globally, and no such additional structure may be imported from a different historical system post hoc.

This raises the evidential status of the **mechanism class**, not of any proposed plaintext, semantic assignment, author, or provenance.

## 8. Stopping point / next discriminating test

Do **not** hybridize Matteo with Ragona/Bartolomeo to repair the failures.

The next clean test, if pursued, is source-independent and falsifiable: compare the K=2 artificial-word output and Voynich under the already existing RF/STA/AAA **component/family and positional** metrics, especially whether any source-faithful selection/ordering principle attested *within Matteo itself* can produce the right-edge constraint. If Matteo supplies no such ordering rule for artificial words, the literal surface route should stop at PARTIAL TRANSFER.

A second independent route is to recover Bartolomeo's actual 100 codewords and use them as an untouched historical holdout/control, not as a fitted generator.

## 9. Compute closeout

No GPU jobs were used. Three short CPU jobs plus one initial connectivity check were launched and terminated normally. Final Hugging Face job check reports **no running jobs**.