# Voynich as compound notation — lossless MDL and historical calibration v0.3

**Date:** 2026-07-24  
**Status:** completed exploratory falsification phase  
**Branch:** `experiment/voynich-notation-falsification-v0.2-20260724`  
**Scope:** structural architecture only; no semantic decipherment or identification as music

## 1. Why this phase was necessary

The v0.2 programme established a small held-out sequence advantage for an HMM over a parameter-matched unordered latent mixture. It then stopped at the stated bottleneck: a lossless, description-length-neutral segmentation comparison and historical notation calibration.

That was not a scientific stopping point. This phase implements both.

## 2. Audit correction to v0.2

The v0.2 HMM used the canonical field `m_core`. The tuple

`prefix + gallows + m_core + suffix`

fails to reconstruct the observed token on **7,534 of 37,465 occurrences**. By contrast,

`prefix + gallows + core + suffix`

reconstructs all 37,465 tokens exactly.

Consequences:

- the v0.2 **absolute values labelled as token codelengths were not lossless token codelengths** and must not be used as such;
- the HMM-minus-IID sequence contrast remains valid as a comparison on the same derived control representation, because the identical core term cancels between the two models;
- the claim that P70 had the best absolute token codelength is withdrawn and replaced by the lossless tests below.

## 3. Lossless two-part / prequential MDL contest

### 3.1 Code construction

Every candidate reconstructs the original token exactly. Each slot vocabulary is transmitted using a KT character code. Token occurrences are then encoded under integrated Dirichlet categorical models. This charges:

1. slot dictionaries;
2. boundary decisions, through the separate slot streams;
3. model complexity, through Bayesian marginal likelihood;
4. every observed surface distinction.

The principal candidates were:

- unsegmented token;
- lossless P70: prefix / gallows / surface core / suffix;
- no-suffix parse;
- ch/sh-as-gallows variants;
- shifted and flattened parses;
- fixed-position splits;
- deterministic random boundaries.

Three Dirichlet concentrations, `alpha = 0.1, 0.5, 1.0`, were frozen as sensitivity values.

### 3.2 Independent-slot code

| alpha | Rank 1 | bits/token | Rank 2 | bits/token | P70 margin |
|---:|---|---:|---|---:|---:|
| 0.1 | No suffix | 14.0851 | P70 lossless | 14.0915 | **−0.0064** |
| 0.5 | P70 lossless | 14.3795 | No suffix | 14.3910 | **+0.0114** |
| 1.0 | P70 lossless | 14.6742 | No suffix | 14.7055 | **+0.0313** |

P70 is competitive and ranks first at two of the three priors, but the margin is tiny and reverses at `alpha = 0.1`. This is a near tie, not a decisive MDL selection.

Both P70 and the no-suffix parse materially outperform the unsegmented exact-value code at all three priors. Thus some affix/core factorisation is useful, but a distinct suffix field is not robustly required.

### 3.3 Direct packet-conditioned code

When the richer packet dependencies are charged rather than treated as free:

| alpha | Best model | P70 rank | P70 bits/token |
|---:|---|---:|---:|
| 0.1 | No suffix | 2 | 14.2244 |
| 0.5 | Unsegmented | 3 | 15.3510 |
| 1.0 | Unsegmented | 3 | 15.9997 |

The added prefix–gallows–suffix contexts overfit under moderate and strong complexity penalties. The v0.2 control packet is therefore **predictively real but not a universal minimum-description-length representation**.

### 3.4 Character-level universal code

A lossless KT bigram code favours leaving tokens unsplit:

| Representation | Global bigram bits/token | Section/position bigram bits/token |
|---|---:|---:|
| Unsegmented | **12.8063** | **12.6388** |
| No suffix | 13.4447 | 13.8281 |
| P70 lossless | 14.1681 | 14.8055 |

The extra slot-end symbols cost more than the slot-specific character regularities save. This rejects the strong claim that the four P70 surface slots are intrinsically the best universal code for Voynich word forms.

## 4. Training-fold-only segmentation transfer

A generic probabilistic segmenter was trained only on the training folios. It learned prefix, gallows and suffix inventories and their positional/section-conditioned distributions, then parsed raw held-out tokens without consulting their stored segmentation.

Five section-stratified complete-folio splits were used.

| Candidate | Exact parse, all held-out tokens | Exact parse, unseen token types | Held-out character-code bits/token |
|---|---:|---:|---:|
| Unsegmented | 100.0% | 100.0% | **12.1791 ± 0.0300** |
| No suffix | 96.7% | **97.2%** | 13.0204 ± 0.0304 |
| P70 lossless | **97.5%** | 94.1% | 13.8761 ± 0.0491 |
| Flat no-gallows | 92.5% | 91.7% | 14.1632 ± 0.0486 |

P70 boundaries are highly learnable and transfer to unseen word types. This is positive evidence that P70 captures stable orthographic structure rather than an arbitrary per-token lookup. However, the simpler no-suffix parse transfers better to unseen types and produces a shorter lossless held-out character code.

## 5. Historical notation pilot

### 5.1 Corpus

A pilot corpus of **416 notation packets** was assembled from ten public-domain GABC transcriptions in the ECHOES GABCtoMEI repository:

- five Aquitanian sources;
- five square-notation sources.

The encoded features retain pitch location, contour, graphical modifiers, neume cuts and full pitch-location sequence. The repository explicitly distinguishes neume cuts from word spaces and records shape, stem, liquescence and ligature information.

### 5.2 HMM versus unordered packet classes

| States | Genuine order gain, bits/packet | Shuffled order gain | Positive held-out documents |
|---:|---:|---:|---:|
| 3 | −0.0407 | −0.0379 | 2/10 |
| 4 | **+0.0179** | −0.0663 | 4/10 |
| 6 | −0.0015 | −0.0603 | 4/10 |

The K=4 historical notation model shows a small genuine-order advantage that disappears under shuffling. It is weaker than Voynich's v0.2 control-state signal of about 0.06 bits/token. The result is unstable across K and based on only 416 packets, so it is calibration evidence, not a family match.

## 6. Independent numerical verification

An independent closed-form recomputation of the `alpha = 0.5` integrated marginal likelihood reproduced the main MDL values to below `1.5 × 10^-13` bits/token for:

- unsegmented;
- P70 lossless;
- no suffix;
- flat no-gallows.

The verification script terminates with `PASS`.

## 7. Revised conclusion

The v0.2 formulation was too strong. The defensible result is now:

> Voynichese has stable, learnable affix/core organisation and a small sequence-dependent signal in derived control features. A lossless P70 segmentation is competitive under exact-value MDL, but it is effectively tied with a simpler no-suffix parse and loses to unsegmented text under universal character coding.

Therefore:

- **weakly stateful structure:** retained;
- **stable packet-like orthography:** retained;
- **unique four-field operational packet:** not established;
- **literal musical notation:** not supported;
- **historical notation family resemblance:** unresolved;
- **P70 absolute token-codelength claim from v0.2:** withdrawn.

The notation hypothesis remains viable only in the broad sense of a structured non-linguistic or procedural code. The present evidence does not distinguish that class from highly constrained language, mnemonic production, recipes, tables, cipher machinery or structured pseudo-text.

## 8. Next evidential gate

The next gate is not another Voynich-only fit. It is a larger blind historical-family classifier using complete GABC/MEI neume corpora and machine-readable German organ/lute tablature, with matched language, cipher and procedural controls. The current ten-document historical pilot validates the ingestion route but is underpowered for classification.
