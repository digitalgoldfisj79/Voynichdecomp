# VBM v145: join-compatible reversible codebooks, Currier A and B

**14 October 2026 · Engineering improvement; Voynich surface NO-GO**

## Decision and cautions (read first)

1. **Engineering PASS:** The original v142b's 222 source units can be represented by individually reversible, prefix-free codebooks with locally attested bridge/cell joins. The local one-escape obstruction for Currier A is removed by allowing two escape prefixes in vowel states `i` and `o`. The independently implemented decoder recovers all 68,396 development and 13,232 held-out Hartlieb words in **both** variants exactly.
2. **Surface NO-GO:** The selected B-compatible mapping still produces unsupported character trigrams in **5,000/13,232 held-out words (37.79%)**; A-compatible mapping in **7,104/13,232 (53.69%)**. Real reference text has average word length **5.06 in B** and **4.78 in A**; the synthetic outputs average **7.96 B** and **8.05 A**. This is not a plausible complete Voynich generator.
3. **Not independent Voynich validation:** The B trigram reference and A ZLZB corpus were **used to build the codebooks**, so a reduction in unsupported trigrams against those same inventories cannot be counted as evidence of historical encoding. Hartlieb is a source-disjoint *German text* holdout, not a manuscript holdout.
4. **Do not promote cipher identification:** v144 already constructed an autonomous, plaintext-free transducer equivalent to the synthetic carry mechanism. Nothing here resolves that identifiability barrier. The source language is unestablished.
5. **No formal infeasibility proof for general VBM:** The one-escape A obstruction applies to the **frozen B-derived cell inventory and bridge map** under the specific observed-support requirement. A new inventory, other escape depth, other spatial segmentation or alternative architecture may evade it.

## Frozen inputs and scope

- Source words and 222-unit ordering: original v142b `04_pair_bridge_manifest.json`, 68,396 development German words, 13,232 untouched Hartlieb words. Preserved in `inputs/`.
- B compatibility: original `/VMS_B_REVA_reference.txt` from v143, 21,679 reference words, reverse-atom EVA representation.
- A compatibility: archived `voynich_transcriptions_slim.json` (ZLZB), 10,995 certain lower-case words from `P` loci of the 114 folios labelled `$L=A` in the independent ZL3b-n primary header. All `$L=B` and uncertain `$L` folios excluded. The metadata list is explicit in `build_v145.py`. Reverse EVA compound atoms `cth,ckh,cph,cfh,ch,sh`, not individual characters. A uses a *separate* word inventory and is not forced to fit B's observed trigram distribution.
- No labels are mixed with running text. The corpus classifier is language metadata, not the broad section-name proxy, which would misclassify mixed Herbal folios.
- The A corpus is taken from a derived normalised transcript, whereas B is the earlier author's raw REVA reference. Thus A and B proportions are independent engineering benchmarks, **not a controlled A/B effect comparison**. Source support is derived from all eligible reference text, not held-out folios.

## Experimental architecture

- Preserve the original **five bridge→cell key maps**, the same 222 source units, and one German word per ciphertext word.
- Use only the original 146/222-element prefix-free B-derived cell sets per vowel state; no invented glyphs, bridges or plaintext escape markers.
- For each vowel state, require every direct `bridge + codeword` to have all 2/3-character substrings attested in the relevant reference. An escape code consists of a prefix-free first cell followed by another cell from the same frozen inventory, with its internal join also attested. Codes remain prefix-free, and the same single codebook is used for every occurrence of a source unit in a given vowel state.
- A: one escape for `a,e,u`; two escapes for `i,o`. B: one escape per state. These encode all 222 units per state.
- Fit the code assignment **only on development German** using a fixed source-unit component adjacency graph (7,319 distinct edges, 88,049 transitions); an additive seam-trigram penalty; and a character-length penalty. Codewords are permuted consistently across all words rather than modified for particular source contexts.
- Hungarian minimum-length initialisation; scan λ in {0.4, 0.8, 1.6} and select the development-lowest seam error subject to a fixed mean ciphertext length cap of **8.25 characters**. This cap is 16.1% above the frozen original v142b development mean of 7.106. Selection chooses λ=0.8 for A and λ=0.4 for B. Hartlieb is not examined in selection.
- Standalone trie-based decoder `verify_independent.py` does not import the encoder or optimiser; validates original source words and computes ciphertext SHA-256. Separate directory replay matches every selected result byte-for-byte.

## Unambiguous headline results

| Measure | Frozen v142b | v145 A-compatible | v145 B-compatible |
|---|---:|---:|---:|
| Original German words decoded in development | 68,396 / 68,396 | 68,396 / 68,396 | 68,396 / 68,396 |
| Held-out Hartlieb words decoded | 13,232 / 13,232 | 13,232 / 13,232 | 13,232 / 13,232 |
| Held-out words with unsupported trigrams (against corresponding A/B reference) | A: 10,324 (78.02%); B: 11,745 (88.76%) | **7,104 (53.69%)** | **5,000 (37.79%)** |
| Held-out unsupported trigram *occurrences* | not compared on unified parser | 12,222 | 7,990 |
| Held-out mean synthetic ciphertext word length | **7.089** | **8.054** | **7.959** |
| Mean native reference word length | n/a | **4.783** | **5.061** |
| Mean development output word length | **7.106** | **7.999** | **8.077** |

Deterministic counts are not inference statistics; null SD is inapplicable. The improvement is a codebook-repair outcome, not a statistical estimate of historical cipher likelihood. A and B use nonidentical reference populations.

## Capacity and trade-off

Frozen original direct `bridge + cell` compatibility for the strict A corpus: a=175, e=171, i=73, o=83, u=107. The corresponding B counts are a=207, e=198, i=117, o=98, u=187. All are below the 222 required units. The **maximum** optimistic one-escape capacity for the frozen A i/o inventories is only **192 and 202**, respectively, so one escape is insufficient there even before long-range joins. With two escapes, local code capacities become **311 (i)** and **321 (o)** for the deterministic candidate choice, permitting 222 distinct prefix-free codewords in each.

Relaxing the length penalty yields a misleadingly attractive result: a B assignment with only 1,291/13,232 invalid Hartlieb words (9.76%) but **mean length 13.81**, and an A assignment with 1,770/13,232 (13.38%) but **mean length 14.51**. The system can bury surface incompatibility under long codes. These are **rejected** by the frozen length budget; they remain in the archive as controls for the trade-off.

For the selected frozen codeword sets, independent Hungarian assignment gives minimum development mean lengths of **7.645 B** and **7.892 A**. These are bounds *for these selected codeword sets and source-text inventory*, not universal lower bounds for VBM.

## What this establishes and what it does not

**Established:** Many earlier v142b violations arose from bad codebook assignment, not from information-preservation requirements alone; a substantial fraction can be repaired while retaining an independently reversible compositional encoder. The frozen B-fitted cell inventory also does not transfer freely to A under a one-escape rule.

**Not established:** Any Voynich content has been recovered; an actual Voynich word is generated by this model; the encoder explains the manuscript's word-length, word-position, token vocabulary, line-entry, cross-word CONNECT or image/label constraints; cipher versus autonomous origin. The model still has an extreme lexical-admissibility miss and oversized words. v143/v144 carry mechanisms were not silently imported: this is a v142b codebook feasibility branch, which can be reconnected later only if it clears surface admission.

**Next scientific gate:** Explicitly bound the **glyph-length and full-word support trade-off**, under frozen encoding overhead and decoding, before optimizing more trigrams. If target word lengths near native A/B are impossible within this architecture, return a bounded architectural NO-GO; do not reinterpret better trigram support as Voynich mechanism identification.

## Reproduction

Within the bundled root:

```bash
python analyze.py > /dev/null
python build_v145.py
python length_bounds.py
python length_constrained_solver.py
python verify_independent.py
```

Inputs are included. `analyze.py` rebuilds local capacity/source tables. The `length_constrained_solver.py` reports selected λ from **development** only and then opens Hartlieb. `verify_independent.py` checks all four outputs against separately stored original German sources. A second clean-directory run matched the selected JSON, ciphertexts, and manifests byte-for-byte.