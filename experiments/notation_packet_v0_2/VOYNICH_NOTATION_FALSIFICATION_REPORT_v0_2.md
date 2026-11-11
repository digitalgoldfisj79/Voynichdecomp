# Voynich as compound notation: v0.2 falsification report

**Status:** exploratory structural result; not a decipherment and not a notation-family identification.  
**Data:** 37,465 P70-decomposed tokens; complete folios are the predictive hold-out unit.  
**Frozen primary question:** does actual token order improve prediction beyond an equally expressive unordered latent packet taxonomy?  

## 1. Decision summary

Two pre-specified gates were evaluated.

1. **Sequence gate — PASSED.** A soft HMM beat a parameter-matched IID latent mixture in **10/10** exact held-out cells: five folio splits at each of K=8 and K=12. The HMM advantage was 0.067 ± 0.031 bits/token at K=8 and 0.063 ± 0.020 at K=12. The minimum observed advantage was 0.032 bits/token.

2. **Segmentation-gain gate — FAILED.** P70's mean direct packet gain was 0.519 bits/token, while several alternate deterministic parses produced larger within-parse gains; the largest was 2.840. Therefore the claimed control packet cannot be justified from *gain over each parse's own independent baseline* alone.

A separate five-split predictive-codelength screen points the other way: P70 had the shortest absolute held-out direct-packet codelength of the seven segmentations in every split, averaging 10.306 bits/token versus 11.655 for the next-best alternative (Flat_no_gallows). This is encouraging, but it does not retroactively pass the frozen gain gate. It is also not a clean independent validation of P70 because P70 itself was developed using the corpus. A proper next comparison must learn the segmentation inside each training fold or use two-part MDL that charges the parse description as well as the predictive model.

## 2. Exact model comparison

| Model | Held-out bits/token | Gain vs independent fields | Increment due to order |
|---|---:|---:|---:|
| Direct observed packet chain | 10.306 ± 0.025 | 0.515 ± 0.015 | — |
| IID latent packet classes, K=12 | 10.231 ± 0.039 | 0.590 ± 0.015 | 0 |
| HMM packet states, K=8 | 10.217 ± 0.047 | 0.604 ± 0.027 | 0.067 ± 0.031 |
| HMM packet states, K=12 | **10.168 ± 0.026** | **0.653 ± 0.010** | 0.063 ± 0.020 |

The result revises the v0.1 conclusion. Most predictive structure is still contemporaneous coupling inside a packet, but the sequence contribution is not zero: actual order carries a small, stable additional signal beyond unordered latent classes.

## 3. Calibration on planted notation

The same pipeline was tested on 12 planted stateful notation corpora and 12 planted IID packet corpora. Surface symbols were arbitrarily permuted and control fields were sometimes omitted.

| Planted family | HMM minus IID, bits/token | Aligned state accuracy | ARI |
|---|---:|---:|---:|
| Stateful notation | 0.869 ± 0.251 | 0.892 | 0.836 |
| IID packet classes | -0.059 ± 0.056 | 0.729 | 0.612 |

The HMM/IID contrast therefore behaves in the intended direction. Voynich's order signal (~0.06 bits/token) is much weaker than the deliberately stateful controls (~0.87), so the result supports **weak state dependence**, not a strongly persistent score-like process.

## 4. Transfer tests

### Across manuscript sections

A direct prefix–gallows–suffix packet model trained on eight sections improved prediction in every held-out ninth section. Gains ranged from 0.397 to 0.526 bits/control-token.

### Across Davis scribal hands

Using the published five-hand map only as a post-hoc transfer diagnostic, leaving out each hand produced positive packet gains for all five hands: 0.384 to 0.539 bits/control-token. Within-section cross-hand transfer was positive in all seven available cells. Mixed Rose pages were excluded and f115r was split after line 12.

This argues against the packet coupling being only a section-specific or individual-scribe habit. It does not independently validate the Davis labels.

## 5. Operational predictions

The preceding suffix family improves held-out prediction of the next prefix by 0.092 ± 0.014 bits/pair across five folio splits.

Line-start odds identify candidate positional operators:

- `s`: odds ratio 3.84
- `y`: odds ratio 3.94
- `d`: odds ratio 1.74
- `ch`: odds ratio 0.20
- `sh`: odds ratio 0.46

Thus `s` and `y` are strong onset candidates, while `ch` and `sh` are anti-onset/continuation candidates. These are functional distributional labels, not semantic translations.

## 6. f115r hand-change test

The control-packet model did **not** detect the known scribal change after line 12.

- Best inferred boundary: after line 17, gain 0.0044 bits/token.
- Davis boundary after line 12: gain -0.0303, rank 17/36.
- The page's best change score was only at the 22.7th percentile of matched Stars-page pseudo-boundaries.

This is negative for a hand-specific control code and compatible with a shared system used across hands. It is not positive evidence for notation by itself.

## 7. Revised scientific conclusion

The strongest surviving claim is:

> Voynich tokens admit a compact control/content factorisation in which prefix, gallows and suffix form a transferable packet. Packet order contains a small but reproducible state signal beyond unordered packet taxonomy.

The following stronger claims are not established:

- that the packet fields are the historical fields intended by the scribes;
- that the mechanism is musical rather than pharmaceutical, mnemonic, astronomical, procedural or generative;
- that any individual glyph means mode, duration, pitch, action or target;
- that P70 is uniquely correct under a description-length-neutral segmentation comparison.

The next load-bearing step is a **training-fold-only or two-part MDL segmentation contest** followed by historical-corpus calibration. The historical data sources have been identified, but no historical-corpus score is included in this verdict because their surface-sign atomisation policies still need to be frozen and their machine-readable files imported consistently.

## 8. Verification questions

### Could unordered packet classes explain everything?
No. In the exact run, the HMM beat the matched IID model in all ten cells. The effect is small but sign-consistent.

### Could P70 manufacture the result?
Partly. Alternate segmentations also manufacture strong packet dependence, and the frozen gain gate failed. P70 nevertheless achieved the best absolute codelength in the screen; only a two-part MDL comparison can adjudicate this properly.

### Is the result merely section or hand information?
No. Packet coupling transferred to every held-out section and all five held-out Davis hands, including positive within-section cross-hand cells.

### Does the f115r result validate the states?
No. It is a negative result: the packet model did not locate the scribal boundary.

### Is this evidence of literal music?
No. It is evidence for a weakly stateful compound notation architecture. Domain identification remains open.
