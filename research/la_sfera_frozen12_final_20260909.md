# La Sfera frozen 12-locus textual-affinity test — final closeout

Date: 2026-09-09

## Question

Do the witnesses selected independently by the frozen Voynich f68r visual comparison, especially Laur2 and Yale4, form a distinctive textual branch in *La Sfera*?

The visual selection was frozen before textual inspection. Twelve Book-I loci were then frozen approximately evenly across the pre-existing public variant list. The primary comparison is Laur2, Yale4, Barb4, Urb2. Cap1 is retained only as a sensitivity/control witness because its Vatican IIIF manifest contains duplicate folio labels and therefore was not allowed to alter the primary four-witness test.

## Normalization

The primary distance is **substantive line-state Hamming**, not literal diplomatic-string equality. Case, punctuation, routine spelling and predictable inflection are normalized (e.g. `secol/secolo/secul`, `creator/creatore`, `ciel/cielo`). Lexical insertion/omission, conjunction state, possessive state and comparable substantive readings are retained. This was necessary because several frozen loci were intentionally minor orthographic variants; a literal spelling metric would mostly measure scribal orthography rather than textual-family signal.

## Frozen state matrix

| locus | Laur2 | Yale4 | Barb4 | Urb2 | substantive character |
|---|---:|---:|---:|---:|---|
| 01.01.02 | same | same | same | same | orthographic only |
| 01.04.07 | same | same | same | same | orthographic only |
| 01.06.07 | A | A | B | C | `di` / power-verb configuration |
| 01.11.01 | A | B | A | B | conjunction before *Leo* |
| 01.12.05 | A | B | A | A | possessive `loro/lor` present vs absent |
| 01.14.07 | same | same | same | same | no substantive difference |
| 01.19.03 | A | A | B | B | conjunction absent vs `et/e` present |
| 01.21.08 | A | A | B | B | `ci` absent vs present |
| 01.23.08 | same | same | same | same | spelling/inflection only |
| 01.28.05 | same | same | same | same | no substantive difference |
| 01.31.02 | same | same | same | same | spelling/inflection only |
| 01.36.08 | same | same | same | same | spelling only |

Seven loci are monomorphic after substantive normalization; five are informative.

## Six pairwise distances

Hamming distance over all 12 frozen loci (equivalently over the five informative loci):

| pair | distance |
|---|---:|
| Laur2–Yale4 | **2** |
| Laur2–Barb4 | 3 |
| Laur2–Urb2 | 4 |
| Yale4–Barb4 | 5 |
| Yale4–Urb2 | 4 |
| Barb4–Urb2 | **2** |

Laur2–Yale4 therefore share a real affinity signal, but they are **not the unique closest pair**: Barb4–Urb2 tie them at distance 2.

## Null 1: state-count-preserving locus-wise relabelling

At each informative locus, retain the observed state multiplicities but randomize the witness labels. For any fixed pair, the per-locus mismatch probabilities are:

- 01.06.07, state counts 2/1/1: 5/6
- 01.11.01, state counts 2/2: 2/3
- 01.12.05, state counts 3/1: 1/2
- 01.19.03, state counts 2/2: 2/3
- 01.21.08, state counts 2/2: 2/3

Thus:

- null mean distance = 10/3 = **3.333333**
- null variance = 19/18 = **1.055556**
- null SD = sqrt(19/18) = **1.027402**
- observed Laur2–Yale4 distance = **2**
- effect = null mean − observed = **1.333333**, i.e. **1.298 null SD closer than expectation**
- exact lower-tail P(D <= 2) = 11/54 = **0.203704**

Exact null distribution:

| D | P(D) |
|---:|---:|
| 0 | 0.00308642 |
| 1 | 0.03703704 |
| 2 | 0.16358025 |
| 3 | 0.33950617 |
| 4 | 0.33333333 |
| 5 | 0.12345679 |

**The metric does not resolve Laur2–Yale4 as a special pair.**

## Null 2: observed pair-label control

Treat the six observed pair distances `{2, 2, 3, 4, 4, 5}` as the finite pair-label null. For a randomly labelled witness pair:

- null mean = **3.333333**
- population null SD = **1.105542**
- Laur2–Yale4 effect = **1.333333 = 1.206 null SD closer**
- inclusive lower-tail rank probability = 2/6 = **0.333333**, because Laur2–Yale4 tie Barb4–Urb2 for the minimum.

Again, **the metric does not resolve this**.

## Interpretation

The original pilot signal was not imaginary. Laur2 and Yale4 agree at multiple independently frozen substantive characters, notably the conjunction omission at 01.19.03 and the `ci` omission at 01.21.08, alongside the distinctive 01.06.07 configuration. However, the complete frozen packet shows that this affinity is non-unique. At other informative loci Laur2 groups with Barb4, while Yale4 groups with Urb2; Barb4 and Urb2 themselves are equally close overall.

The appropriate status is therefore:

`TEXTUAL_AFFINITY_SIGNAL_REAL_BUT_NONUNIQUE__METRIC_DOES_NOT_RESOLVE_BRANCH`

This closes the La Sfera textual-family gate. The data support shared textual tradition / crossing affinities, not a uniquely identified Laur2–Yale4 branch and not an identifiable exemplar for Voynich f68r.

## Geographic consequence

**No Florence gate is opened.** Even a unique textual branch would not by itself localize manufacture; this test does not even reach that threshold. The result changes neither the A/B/C geography firewall nor the current conclusion that La Sfera is a source/iconographic comparator rather than a proven geographic locator for MS408.

## Reproducibility / exclusions

- Visual witness selection frozen before textual inspection.
- Twelve loci frozen before target-witness readings.
- No witness substituted after seeing its textual state.
- Full-page recovery was used after proportional crops proved unreliable.
- Primary four-witness result excludes Cap1 from branch inference because the Vatican manifest's duplicate folio labels create an image-selection ambiguity; Cap1 cannot rescue or overturn the primary result post hoc.
- No geographic inference is made from present repository location.

Final decision: **close this branch unless a genuinely independent stemmatic dataset or hard provenance/production edge appears. Do not expand generic La Sfera witness searching further.**
