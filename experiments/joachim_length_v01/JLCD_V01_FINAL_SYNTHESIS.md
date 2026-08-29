# JLCD v0.1 — final synthesis

## RETRACTED FINDINGS

None from the primary JLCD programme. The previous WLCP finding of a generic adjacent/Markov length-transition grammar remains retracted after section/position conditioning; nothing here restores it.

## ENDPOINT

**JLCD-0: Joachim's proposed length-conditioned e/i-counter mechanism is not supported by the registered tests.**

Primary corpus: ZL3b running text, 33,161 tokens across 207 folios. Source SHA-256: `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`.

### 1. Necessary same-core length/context condition

After stripping operational e/i counter units, holding the remaining core fixed, and conditioning on Currier, section, line-position bucket and line-length bin, total token length does not reproducibly predict external context.

EVA: full effect 0.003490 bits/occurrence, null SD 0.002799, z=1.25; Currier A z=1.15; Currier B z=0.94.

Raw characters: full effect 0.001787, null SD 0.002672, z=0.67; A z=0.50; B z=0.54.

All are below the registered 2-SD resolution threshold.

### 2. Proposed short/long regime boundary

No replicated 3/4 boundary is present. The nominal full-EVA k=3 statistic reached z=2.07 but failed the corrected threshold scan (FWER p=0.2156), failed Currier replication, and failed the raw-character representation. k=4 is unresolved throughout.

### 3. e/i specificity

The e/i stripped-core effect is not exceptional among frequent-unit controls: effect 0.003162 bits/occurrence, null SD 0.002806, z=1.13, rank 15/28. Several other unit-pairs produce substantially larger effects; a/d ranks first at effect 0.017133, null SD 0.003473, z=4.93.

### 4. Exact one-unit near-neighbour bound

Because deleting all e/i could be too aggressive if only *additional* e/i act as counters, a narrower post-primary bound was frozen and run on exact attested one-unit insertion pairs.

11,287 one-unit pairs were discovered; 359 disjoint e/i pairs and 481 disjoint non-e/i pairs were selected.

Exact e/i pairs remain unresolved:
- full EVA: effect 0.002244 bits/occurrence; null SD 0.002181; z=1.03.
- A EVA: effect 0.006821; null SD 0.003638; z=1.88.
- B EVA: effect 0.001055; null SD 0.002525; z=0.42.
- full raw-char: effect 0.000794; null SD 0.002155; z=0.37.

The proposed boundary steps also fail replication. For 3→4, full EVA z=1.58, A z=2.39, B z=0.33; raw-character full/A/B z=0.92/0.14/0.95. For 4→5, all EVA and raw-character tests are below 2 SD.

By contrast, non-e/i one-unit insertion pairs show a strong generic near-neighbour/context effect: full EVA effect 0.010232, null SD 0.001810, z=5.65; A z=5.04; B z=3.45; raw-character full z=5.63, A z=4.69, B z=3.91.

This reverses the specific prediction that e/i insertions should be unusually context-disruptive if they function as length counters moving a core into a different lookup list.

### 5. Power bound

The exact-pair test has low sensitivity to a moderate injected context bias: recovery above null+2SD was only 6% at injected OR=1.5 and 16% at OR=2.0 in EVA (4% and 10% in raw characters). Therefore the null result does **not** exclude every subtle e/i-dependent mechanism. It does exclude the claim that a strong, readily observable e/i-specific length-indexed codebook signal is already visible at the scale and form proposed.

### 6. ked / keed example

The supplied example cannot independently carry the claim in ZL running text: `ked` occurs once and `keed` three times. Any contextual interpretation of this pair alone is underpowered.

## Interpretation

The data contain genuine word-form/position structure, as WLCP already showed. But when Joachim's mechanism is made discriminating — same operational core, e/i-specific additions, proposed 3/4 threshold, matched Currier/section/position controls — its predictions fail. More importantly, generic non-e/i one-unit variants show a much stronger contextual separation than e/i variants.

The best current explanation is therefore **generic structured morphology / positional variant selection**, not an e/i counter system that indexes different length-specific cipher lists.

This does not disprove every conceivable length-sensitive cipher. A materially different version would need to specify in advance which e/i are counters, how the core is identified, where boundaries fall, and what external-context prediction follows. Those frozen rules could then be tested without post-hoc reclassification.
