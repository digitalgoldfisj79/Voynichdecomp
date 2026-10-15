# v144: bridge-orbit signature and observational equivalence
**14 October 2026 · Status: synthetic mechanism detectable, plaintext attribution NOT IDENTIFIABLE**

## RETRACTIONS AND NEGATIVE RESULTS (maintained at file top)

1. **Retracted as a plaintext-driven finding:** The predecessor-bridge/orbit association, even when enormous against the v142b and state-free controls, does **not** identify hidden plaintext-driven carry. A purely ciphertext-reading transducer reproduced every v143 output byte using the original v142b ciphertext and frozen manifest. The same associative signature therefore arises without accessing any plaintext during cross-word state propagation.
2. Earlier v143 visible-tail-proxy evidence is not admitted: last-three-character proxy is neither guaranteed to encode the latent state nor more diagnostic than the unchanged baseline.
3. v142b/v143 surface compatibility NO-GO persists: this test is run **only on synthetic ciphertext**, not the manuscript, and does not overturn the severe trigram violations reported previously.
4. Neither Currier A nor B has been scored with this instrument. Without an independently justified manuscript lexer and free-standing choice-of-word model, attempting the test on either class would import the German-trained codebook and produce circular results.

## Initial hypothesis and qualification

Candidate: cell-code permutations preserve an ordered sequence of bridge-pair literals and cell (first-glyph, length) classes. In the frozen v143 grammar, the previous word's last *bridge pair* exactly encodes the six-way carry, and this should predict the choice of the following word's first cell, after conditioning on that next word's invariant structural skeleton. This is a ciphertext-only *feature* test, but its lexer and bridge vocabulary use the known synthetic manifest; it is **not key-independent** for historical Voynich.

Nulls: (a) unchanged v142b sharing the same German source, (b) index-scheduled statefree permutations using the same codebook, (c) 80 matched shuffles of previous-bridge labels among identical current-word skeletons and 2,000-word blocks. All arms share token positions and skeletons. Test data: 13,232 held-out Hartlieb ciphertext words. No plaintext or decoder calls enter feature extraction, prediction or scoring.

## Quantitative result

| Arm | Held-out gain bits/eligible word | Gain relative to matched permutation mean | Permutation SD | Effect/SD |
|---|---:|---:|---:|---:|
| v142b unchanged | +0.063628 | +0.132832 | 0.005235 | 25.4 |
| v143 carry | +1.039429 | +1.910881 | 0.020554 | 93.0 |
| statefree | -0.016954 | +0.052323 | 0.004895 | 10.7 |

N=8,036 eligible identical positions; 99.35% permutable under the chosen block/shell null; five folds all have carry stronger than both negative controls. The *raw* held-out carry-v142b gain difference is +0.975801 bits/word, while the *null-centred* difference is +1.778049 bits/word and paired-null SD 0.020789 (85.5 SD). Raw carry-statefree is +1.056383, null-centred +1.858557, paired-null SD 0.021453 (86.6 SD). These are synthetic conditional-permutation effect sizes, **not historical confidence**.

## Stronger control: the exact autonomous reconstruction

A second independent implementation generated the carry ciphertext from v142b ciphertext plus the frozen v143 manifest, without reading normalised German during the generation pass. It updated the carry state using the *previous already-emitted ciphertext word*, not the previous source text and not even the previous baseline input. The independent source-text check was performed only after generation.

| Corpus | Ciphertext words reconstructed | Exact byte equivalence | SHA-256 |
|---|---:|---|---|
| Development | 68,396 | True | `c41af1c1a65f8a84bbfd8fb03c32135d63a7b0ab336bf2ed70d7190491d468fb` |
| Hartlieb holdout | 13,232 | True | `587ab349b448ca2c31dfcdca76cb37dc5f66c79dece8c4226ffb31e9a7af29d8` |

**Total: 81,628/81,628 words and both complete ciphertext files reproduced byte-for-byte;** deterministic equality, so a null SD is not applicable.

The mathematical reason is straightforward. For all valid inputs under the frozen grammar, every plaintext vowel is manifested in a bridge pair untouched by the subsequent code-cell rotation. Let B(c) return the vowel of the final bridge pair in ciphertext word c; let P_s permute the permitted code cells using state s. If E0(w) is v142b's encoding, then:

`v143(w_i, previous) = P_{B(c_(i-1))}(E0(w_i))`

and `B(P_s(E0(w))) = B(E0(w))`. Thus the carry update is **observably redundant with the previous emitted word**, given the known synthetic grammar. An autonomous surface grammar can have precisely the same ciphertext output distribution and exact sequence.

This is not a refutation of language encoded in the *base words*. It is an identifiability proof for the **additional historical carry attribution**. A third party cannot tell whether the last bridge originally denoted a plaintext vowel or an autonomous state symbol by examining that stream alone.

## Audit order

1. **Circularity:** v142/v143 was designed with nine bridge pairs; qualification only on untouched Hartlieb synthetic output, not against Voynich. Any VMS bridge interpretation would require independent validation.
2. **Leakage:** score reads ciphertext and frozen lexer, no plaintext or encoder-event logs. The autonomous reconstruction separately proves exact equivalence and does not read plaintext during generation. The *lexing table* is known; don't call this key-free.
3. **Confounds:** source word order/semantic dependencies can drive baseline conditional effects. Strongest non-carry control is the exact autonomous replica, which removes historical source attribution altogether.
4. **Matched nulls:** constrained permutations preserve current skeleton, 2,000-word block membership, sampled positions, outcomes, and current marginal bridge state; 80 null replicates.
5. **Fairness:** same text/source positions, held-out folds, codebook and skeleton in three arms; four-arm equivalent autonomy control yields exact identity rather than a statistical difference.
6. **Measurement degeneracy:** within-shell shuffled null had 99.35% permutable observations and positive nonzero SD; but the exact equivalent control is necessarily indistinguishable.
7. **Representation dependence:** tested v142b/v143 REVA-inspired synthetic codeword lexer. No direct EVA/REVA A/B extension; inherited B-only surface failure still stands.
8. **Decision fragility:** synthetic discriminator crosses the precommitted two-null-SD and five-fold gates, but carries no unique plaintext attribution once exact-autonomous control is admitted. No posthoc metric adjustment.
9. **Audit completeness:** frozen protocols, original codebook, all source files, scripts, results, sha hashes and atomic pickle checkpoints provided. Prior retractions above remain live.

## Decision and next genuine opportunity

**GO:** An economical *surface-state permutation* can carry context-dependent word variants without additional glyphs. Its observable is a structured, low-cardinality transformation orbit, not crude last-letter dependence.

**NO-GO:** Claim that the observed orbit implies plaintext-driven carry. An exact autonomous transducer explains it equally.

**NEXT INSTRUMENT, not yet admitted:** Identify candidate word orbits from the separately established, section-specific FORM representation (not v142's German-fitted lexer). Freeze orbit-equivalence rules on training quires, then on untouched quires compare a **shared low-complexity cyclic state-permutation** to (i) matched whole-word Markov CONNECT; (ii) independent state-conditioned arbitrary substitutions with the same state count and entropy; (iii) an autonomous process using the same observed states; and (iv) label/state shuffles matched for line, section, word frequencies and token geometry. Fit Herbal A, Pharma A, Herbal B, Biological B and Stars B in their own sections and verify A↔B transfer only after within-section qualification. This would assess the structure of selection rules, **not prove source-language encoding**. Until an independently derived lexer/orbit exists, this experiment is unqualified and should not be run.
