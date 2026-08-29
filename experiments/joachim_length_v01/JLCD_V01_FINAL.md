# JLCD v0.1 — FINAL BOUNDED SYNTHESIS

## RETRACTED FINDINGS

None from the preregistered JLCD run.

A post-hoc matched e/i-vs-non-e/i specificity contrast was attempted but is **not promoted** because, despite one-to-one lexical-pair matching, the strict context-matched eligible support was badly imbalanced (e/i 9,048 eligible occurrences vs control 619). Its negative contrast therefore remains unresolved rather than evidence against e/i.

## ENDPOINT

**JLCD-0 under the preregistered decision rule: Joachim’s specific e/i-counter + 3/4 length-class mechanism is not supported.**

This is not equivalent to proving that token length never matters. The near-neighbour external-context test has low recovery for moderate injected effects, so moderate context dependence cannot be excluded.

## 1. Primary e/i-stripped-core test

Question: after holding the apparent non-e/i core, Currier, section, line-position bucket and line-length bin fixed, does total token length predict immediate external context?

EVA:

- Full corpus: **the metric does not resolve this — effect = 0.003490 bits/occurrence; matched-null SD = 0.002799; z = 1.25** (7,095 eligible occurrences, 316 cores).
- Currier A: **the metric does not resolve this — effect = 0.008043; matched-null SD = 0.006977; z = 1.15**.
- Currier B: **the metric does not resolve this — effect = 0.002712; matched-null SD = 0.002879; z = 0.94**.

Raw-character representation:

- Full corpus: **the metric does not resolve this — effect = 0.001787; matched-null SD = 0.002672; z = 0.67**.
- Currier A: **the metric does not resolve this — effect = 0.003666; matched-null SD = 0.007352; z = 0.50**.
- Currier B: **the metric does not resolve this — effect = 0.001545; matched-null SD = 0.002886; z = 0.54**.

Primary T1 therefore fails in both representations and both Currier replications.

## 2. Claimed 3/4-glyph regime boundary

The fixed threshold scan tested k=2..8 with joint max-|z| familywise correction. Joachim’s proposed k=3 or k=4 split did not replicate.

At k=3:

- Full EVA: effect = 0.009488; null SD = 0.004591; z = 2.07, but familywise p = 0.2156.
- Currier A EVA: **the metric does not resolve this — effect = 0.006174; null SD = 0.005450; z = 1.13**.
- Currier B EVA: **the metric does not resolve this — effect = 0.009646; null SD = 0.005239; z = 1.84**.
- Full raw-character: **the metric does not resolve this — effect = 0.014465; null SD = 0.008448; z = 1.71**.

At k=4 every full/A/B test is below 2 null SD in both representations.

The strongest full-EVA threshold was k=5, not k=3/4. No registered 3/4 change-point survives correction and replication.

## 3. e/i specificity

When the same stripped-core test was repeated for every pair among the eight most frequent EVA units (`o,e,y,a,d,i,ch,l`), `{e,i}` ranked **15th of 28** by bias-corrected context effect.

For e/i: **the metric does not resolve this — effect = 0.003162 bits/occurrence; matched-null SD = 0.002806; z = 1.13**; eligible = 7,095. The median non-e/i control support was 3,819, so e/i was within the preregistered 2× sample-size bound, but it did not reach the top-quartile effect criterion.

Several other strip pairs produced larger effects, including a/d (effect 0.017133; null SD 0.003473; z 4.93), d/i (0.011899; 0.004268; z 2.79) and i/o (0.008358; 0.003438; z 2.43). This does not identify those pairs as cipher operators; it shows only that e/i is not uniquely exceptional under this test.

The later one-to-one matched specificity contrast is not promoted because its strict eligible support became highly imbalanced after contextual matching.

## 4. Narrower exact one-e/i-insertion bound

To avoid the possible over-aggressive assumption that *all* e/i are counters, a post-primary bound compared exact attested near-neighbours differing by one inserted `e` or `i` only. A deterministic disjoint set of 359 e/i pairs was used.

EVA external-context test:

- Full: **the metric does not resolve this — effect = 0.002244 bits/occurrence; null SD = 0.002181; z = 1.03**; eligible 9,337.
- Currier A: **the metric does not resolve this — effect = 0.006821; null SD = 0.003638; z = 1.88**; eligible 2,141.
- Currier B: **the metric does not resolve this — effect = 0.001055; null SD = 0.002525; z = 0.42**; eligible 7,196.

Raw-character results are weaker (full z 0.37, A z -0.21, B z 0.43).

The exact 3→4 and 4→5 e/i insertion subsets also fail full/A/B replication in both representations. The sole isolated >2 result is Currier A EVA at 3→4 (effect 0.009920; null SD 0.004151; z 2.39), which fails the full/B and representation replications and is not promoted.

### Power bound

This exact-pair CMI test is not highly sensitive to moderate synthetic context effects. With variant counts preserved inside every matched stratum, an injected OR=1.5 context preference was recovered only 6% of the time in EVA; OR=2.0 only 16%. Raw-character recovery was 4% and 10% respectively.

Therefore the correct inference is **no detected effect**, not “moderate e/i context dependence is impossible.”

## 5. Does Joachim’s e/i idea explain the WLCP positional result?

This is the most useful connection to the preceding programme.

WLCP found that, within related short/long forms generally, the longer member is preferentially line-initial. Decomposing exact insertion families by inserted unit gives:

### e/i insertion families

- Full: **effect = 0.526189 log-odds; matched-null SD = 0.103077; z = 5.10; observed OR = 1.693**.
- Currier A: **the metric does not resolve this — effect = 0.246233; null SD = 0.212704; z = 1.16; OR = 1.277**.
- Currier B: **effect = 0.627471; null SD = 0.126135; z = 4.97; OR = 1.871**.

### non-e/i insertion families

- Full: **effect = 0.266926 log-odds; null SD = 0.081195; z = 3.29; OR = 1.307**.
- Currier A: **effect = 0.342785; null SD = 0.145889; z = 2.35; OR = 1.411**.
- Currier B: **effect = 0.225586; null SD = 0.106156; z = 2.13; OR = 1.253**.

Thus e/i lengthening clearly participates in the positional phenomenon in the full corpus and especially Currier B, but it **does not reproduce the cross-Currier rule** because Currier A does not resolve it. Conversely, non-e/i additions reproduce the positional rule in full, A and B.

The explicit test distinguishing “e/i counters explain our WLCP result” from “the rule is a broader morphological/positional phenomenon” is A/B replication of the same within-line-swap positional odds effect. e/i fails that test; non-e/i passes it.

## 6. Joachim’s `ked` / `keed` example

In the frozen running-text corpus, exact `ked` occurs once and exact `keed` three times. They occur in different section mixtures. Four occurrences cannot support a lexical-context inference, so this example is statistically non-diagnostic.

## Final interpretation

The data do not support Joachim’s specific proposed architecture as stated:

- no replicated independent e/i-stripped-core length/context effect;
- no replicated/corrected 3/4 length-class change-point;
- e/i is not exceptional among frequent stripped-unit pairs;
- exact one-e/i near-neighbours do not show a replicated external-context effect;
- e/i additions participate in the known positional short/long-form rule, but non-e/i additions do too, and only the latter replicate that rule across both Currier systems.

The strongest defensible statement is therefore:

> **Voynich token length is structurally entangled with variant selection and line position, but the evidence does not resolve Joachim’s proposed e/i counter / length-indexed codebook mechanism.**

This leaves open weaker or differently specified length-conditioned mechanisms, especially because the exact-neighbour external-context test has low power for moderate effects. A stronger Joachim test would require him to freeze which e/i occurrences are counters, the exact length bins, and a predicted mapping from those bins to observable context/decoding behaviour before looking at further data.

## Scope boundary

No result here validates or tests the claimed `ked = nd` plaintext value, “vowel bridge”, “missing 20 percent”, or proposed historical construction sequence, because the supplied post does not specify those mechanisms sufficiently to generate independent held-out predictions.
