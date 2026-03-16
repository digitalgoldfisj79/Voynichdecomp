# Supplement S10: Negative Results

## Overview

This supplement documents hypotheses that were tested and rejected during the
development of the cipher architecture. The method's limits are as informative
as its successes. All tests used the same enriched_records dataset and scoring
battery described in the main text.

---

## 1. f17v Fennel False Positive (§9.1)

**Hypothesis:** The CI fennel (feniculum) entry's CV profile should correlate
most strongly with f17v, which some researchers have tentatively identified as
a fennel folio.

**Test:** Computed the CI fennel entry's distinctive CV profile and ranked all
48 HA folios by correlation.

**Result:** f17v ranks **1st out of 48** (r = 0.365, p = 0.0008 from 5,000
random CI passages of same length).

**Falsification:** f17v's illustration depicts a broad-leaved plant with berry
clusters, tentatively identified as *Tamus communis* (Zandbergen: ELV tamus,
ThP smilax/tamus communis). This is unambiguously not fennel (feathery leaves,
yellow umbels).

**Interpretation:** The statistical correlation reflects shared pharmaceutical
vocabulary (fe-initial words like *febribus*, *fel*) rather than plant identity.
This establishes the principal limit: CV profile matching produces false positives
when vocabulary overlap is driven by shared ingredients or shared therapeutic
vocabulary rather than plant names.

---

## 2. Two-Syllable Encoding (§9.2)

**Hypothesis:** Consecutive FC tokens between EC anchors might encode consecutive
syllables of the same Latin word.

**Test:** Measured match rate between VMS FC doublets and CI two-syllable word CV
pairs.

**Result:** 45.6% match, versus null models of 47.5% (random CV pairs, p = 0.73)
and 49.8% (shuffled VMS FC stream, p = 0.88).

**Conclusion:** Match rate is at or below chance. Each FC token independently
encodes one word's first CV. Consecutive tokens are independent words, not
consecutive syllables. The VMS is a word-for-word cipher, not a syllable-for-syllable
cipher.

---

## 3. Nomenclator Extension (§9 context)

**Hypothesis:** Extending the nomenclator beyond the 12 currently assigned function
words might improve the bigram correlation.

**Test:** Greedy search continuing past 12 assignments.

**Result:** Every additional word beyond the 12 tested shows zero improvement
(Δr < 0.002) when reassigned from its vowel-heuristic family. The R-family
(e-initial: est, eius, per, ex, sed) and BARE-family (u-initial: quod, ut, qui)
words are all correctly placed by the heuristic.

**Conclusion:** The nomenclator is complete at 12 entries. The cipher designer
only needed nomenclator entries for words whose first vowel did not match their
intended family. All e-initial and u-initial function words are correctly routed
by the grid rule itself.

---

## 4. Per-Section Keywords (§9 context)

**Hypothesis:** Different manuscript sections might use different babuini grid
keywords, explaining the cross-section vocabulary differences.

**Test:** 278 keywords evaluated through the forward cipher across all 9 sections.

**Result:** All scored 51–53/84. No keyword produces section-specific improvement.
Cross-section variation is explained by source text differences, not keyword changes.

**Conclusion:** A single keyword (or no keyword) applies to the entire manuscript.
Section-level vocabulary differences reflect different source text content, not
different cipher configurations.

---

## 5. Circa Instans as Direct Source Text (§9 context)

**Hypothesis:** The Wellcome MS 624 Circa Instans might be the specific text
being enciphered.

**Test:** Forward cipher using CI directly as source text.

**Result:** CI performs well as a genre proxy but the autocorrelation metrics
(which measure topical clustering) never pass. The specific vocabulary ordering
in CI does not match the VMS's folio-level vocabulary patterns.

**Conclusion:** The source text is CI-tradition pharmaceutical Latin but not
this specific CI manuscript. The source is an unidentified text from the same
tradition.

---

## 6. Pharmaceutical Vocabulary Direct Assignments (§9 context)

**Hypothesis:** Specific pharmaceutical terms (herba, radix, aqua, etc.) might
be directly identifiable in the VMS through their grid-cell frequencies.

**Test:** Tested 5 high-frequency pharmaceutical terms for distinctive CV profiles
on specific folios.

**Result:** 0/5 produced significant results. Pharmaceutical content words are
too evenly distributed across folios (appearing on most folios in similar proportions)
to produce folio-specific enrichments.

**Conclusion:** The CV reader identifies folio-distinctive vocabulary, not
corpus-common vocabulary. Generic pharmaceutical terms like *herba* appear on
nearly every folio and therefore cannot distinguish any single folio.

---

## 7. Hybrid Engine Approaches (development context)

**Hypothesis:** Combining the forward cipher with Gen-SP (the circular slot-pair
generator from Paper 1) might improve the 84-metric score.

**Test:** Multiple architectures tested, including bolting Gen-SP onto v11 output,
using Gen-SP for FC tokens only, and mixing Gen-SP and v11 outputs.

**Result:** All hybrid approaches degraded scores. Gen-SP and v11 optimise different
distributional properties; combining them produces output that satisfies neither
target.

**Conclusion:** The forward cipher and the grammar generator are complementary
analyses, not composable components.

---

## Summary

| Hypothesis | Result | Lesson |
|-----------|--------|--------|
| f17v = fennel | False positive | CV matching hits shared vocabulary, not plant identity |
| Two-syllable encoding | Below chance | Word-for-word cipher, not syllable |
| Extended nomenclator | No improvement | 12 entries is complete |
| Per-section keywords | All equal | One keyword for whole MS |
| CI as direct source | Autocorrelation fails | Right tradition, wrong manuscript |
| Pharma vocab assignments | 0/5 | Common words can't distinguish folios |
| Hybrid engines | Degraded scores | Analysis tools don't compose as generators |
