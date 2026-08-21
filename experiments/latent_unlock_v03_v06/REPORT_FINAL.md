# Latent surface inversion / unlock programme — final closeout
Date: 2026-08-21

## Question
Can the demonstrated local head-selection / boundary-production layer be separated from meaning-bearing morphology in an information-preserving way, and does doing so expose a recoverable lexical/cipher stream?

## v0.3 — unsupervised latent transducer
Model: each observed token is either latent `z` or a one-glyph context-sensitive surface marker prepended to `z`; EM is fitted out-of-document with no target-language information.

Development MHG calibrated, but frozen transfer failed: the Latin and especially small volgare controls collapsed to the no-surface local optimum. Voynich remained sealed.

## v0.4 — source-internal seeded EM
Only change: initialise the surface channel from recurrent prefix-pair/context counts. MHG and Latin recovered the planted layer, but the 3.5k-token volgare control still collapsed to the null solution. Voynich remained sealed.

## v0.5 — frozen structural surface detector
A language-blind detector was trained only on known-answer synthetic Ortolf. Features are recurrence, prefix-pair support and immediate-boundary context statistics computed out-of-document. Probability threshold was frozen at 0.98; no target-language features or Voynich information entered training.

Development / frozen validation:

| corpus | precision | recall | exact improvement | ordinary-text false change |
|---|---:|---:|---:|---:|
| Ortolf MHG | 98.88% | 67.82% | +26.65 pp | 2.22% |
| Admonter MHG | 98.84% | 72.49% | +28.74 pp | 1.15% |
| Antidotarium Latin | 98.86% | 65.26% | +26.31 pp | 2.17% |
| Antidotarium volgare | 98.69% | 52.84% | +21.12 pp | 2.78% |

All frozen gates passed, so the Voynich arm was opened.

Voynich transform:
- 34,415 tokens in the selected ZL corpus frame.
- 29,617 noninitial eligible events.
- 2,670 high-posterior surface heads changed: 9.015% of eligible events.
- types 7,177 -> 6,679.
- mean token length 4.9941 -> 4.9165.
- Original tokens and full posterior probabilities are retained in `v05/VOYNICH_POSTERIORS.csv`; the canonical map is therefore reversible as an analysis object.

## v0.6 — end-to-end lexical unlock test
A substitution-class quadratic-assignment solver was frozen on within-token bigrams plus start/end/unigram profiles, fitted on four-fifths of source documents and scored for exact target-lexicon recovery on the held-out fifth.

### Known-answer gate: separate Breslau MHG ciphertext
The control uses an untouched MHG corpus, a fresh random monoalphabetic key, and the same planted context-sensitive surface layer.

| representation | key accuracy | exact 7+ recovery | all held-out recovery |
|---|---:|---:|---:|
| true latent ciphertext | 92.31% | 99.45% | 99.94% |
| surface-corrupted | 84.62% | 38.71% | 61.00% |
| v0.5 residual | 88.46% | 79.05% | 87.81% |

The end-to-end instrument therefore passes: inversion restores most of the long-form recoverability destroyed by the planted surface layer.

### Voynich — full frozen solver

MHG target:

| length | raw | residual |
|---|---:|---:|
| 2–3 | 80.35% | 78.62% |
| 4 | 6.28% | 19.23% |
| 5 | 2.04% | 0.52% |
| 6 | 0.18% | 0.09% |
| 7+ | **0.00%** | **0.00%** |
| all | 18.79% | 20.80% |

Latin target:

| length | raw | residual |
|---|---:|---:|
| 2–3 | 39.86% | 25.74% |
| 4 | 5.99% | 11.63% |
| 5 | 0.39% | 2.75% |
| 6 | 0.09% | 0.63% |
| 7+ | **0.00%** | **0.00% |
| all | 9.66% | 8.56% |

Binding unlock criterion fails immediately: after an inverse that demonstrably restores long-form recovery on known-answer ciphertext, Voynich still has zero exact recovery at length 7+ against both MHG and Latin.

### Matched-removal sensitivity
Thirty random controls remove the same number of first glyphs, matched by section and original token length. To make the 30-arm sensitivity tractable, all observed and null arms use the same reduced QAP solver (3 restarts x 1000 steps); these absolute values are not mixed with the full-solver table above.

For MHG, residual minus matched-null in null SD units:
- all coverage: -0.66 SD
- length 2–3: -0.78 SD
- length 4: -0.21 SD
- length 5: -0.68 SD
- length 6: +2.33 SD (isolated nominal cell)
- length 7+: -0.26 SD

The apparent short-form changes are therefore not a privileged lexical recovery signature. The isolated length-6 cell does not continue to length 7+, is one of several inspected bins, and does not satisfy the preregistered unlock rule.

## Binding conclusion
**NO LEXICAL UNLOCK.**

What has been added to the evidence base:
1. A context-sensitive surface layer can now be separated from genuine morphology on known-answer MHG, Latin and volgare controls with high precision and low false-change rates.
2. The same frozen detector identifies a modest, nontrivial surface-like layer in Voynich (~9% of noninitial tokens).
3. The complete inversion + substitution solver is demonstrably capable of restoring long-word recovery when such a recoverable payload exists beneath the synthetic surface layer.
4. Applying that qualified pipeline to Voynich does not restore long-form recovery: 7+ remains 0% for both MHG and Latin.
5. The modest short-form changes are no better than matched random head removal on the principal sensitivity metrics.

This closes the specific hypothesis that the already-demonstrated local slot/head-production layer is what has been masking an otherwise ordinary substitution-class MHG/Latin lexical stream. It does **not** exclude verbose, many-to-one, transpositional, stateful cipher families, notation systems, or meaning carried jointly by multiple slots; those require a solver that first clears its own known-answer control for that family.
