# HEAD-SELECTION-v0.1 — conditional next-head selection

## Question

Does the immediately preceding token final help select the first EVA character of the next token **after the remainder of that next token is already fixed**, beyond section/Currier/hand/line-position production context?

Primary literal representation: `head = next_token[0]`; `stem = next_token[1:]`. This deliberately matches the supplied `stem_len=1` analysis rather than imposing a linguistic stem definition.

## Calibration

Synthetic no-local-dependence control: -0.011740 ± 0.001477 held-out bits/event.
Synthetic planted local-selector control: +0.055900 ± 0.002268 held-out bits/event.
The instrument therefore distinguishes planted immediate selection from a no-dependence surface on the real covariate skeleton.

## Corpus

28,530 line-internal next-token events from 217 folios and 4,056 lines; 4,298 literal remainder-types before training eligibility filters. The inferential set contains 18,754 held-out events whose stems meet the training-only ≥20 occurrences / ≥3 folios gate.

## Primary result

Baseline (`stem + section/Currier/hand/line-position`): **1.155010 bits/event**.
Add actual immediately preceding final: **1.139540 bits/event**.
Held-out gain: **+0.015471 bits/event**.
Matched within-line final-shuffle null: **+0.003355 ± 0.001396 bits/event** (25 completed permutations; frozen plan was 100; execution-only amendment).
Excess over null: **+0.012116 bits/event = 8.68 null SD**.

All five complete-folio folds are positive: F0 +0.01298, F1 +0.01711, F2 +0.00904, F3 +0.01557, F4 +0.02156.

## Locality / lag control

On the **same 15,677 events** where both lag-1 and lag-2 exist: lag-1 = **+0.016765**, lag-2 = **+0.004758**; adjacency excess = **+0.012007 bits/event**.
On the **same 12,824 events** where both lag-1 and lag-3 exist: lag-1 = **+0.016754**, lag-3 = **+0.001379**; adjacency excess = **+0.015375 bits/event**.
The effect is therefore sharply boundary-local rather than a generic page/line state carried by earlier tokens.

## Smoothing sensitivity

| Dirichlet τ | lag-1 gain | lag-2 gain | lag-1 minus lag-2 |
|---:|---:|---:|---:|
| 5 | -0.024791 | -0.040137 | +0.015346 |
| 10 | -0.001372 | -0.014293 | +0.012921 |
| 20 | +0.015471 | +0.003284 | +0.012186 |
| 50 | +0.031245 | +0.018046 | +0.013199 |

At τ=5 and 10 the richer model overfits in absolute held-out codelength, so the broad claim “previous final always improves prediction under any smoothing” is **not** supported. However the **lag-1 advantage over the otherwise identical lag-2 model remains +0.012–0.015 bits/event at every τ**, which is the more specific locality claim.

## Supplied highlight families after controls

| remainder/stem | held-out n | lag-1 gain | lag-2 gain | adjacency excess |
|---|---:|---:|---:|---:|
| `aiin` | 901 | +0.01028 | -0.00729 | +0.01756 |
| `l` | 718 | +0.02678 | -0.00043 | +0.02721 |
| `ain` | 246 | -0.00104 | -0.00965 | +0.00862 |
| `kaiin` | 254 | +0.01650 | -0.02218 | +0.03868 |
| `ly` | 72 | +0.01929 | +0.01056 | +0.00872 |
| `ar` | 389 | -0.03464 | -0.02207 | -0.01257 |
| `eedy` | 94 | +0.04904 | +0.02595 | +0.02309 |
| `r` | 611 | +0.03630 | -0.01008 | +0.04638 |
| `cheey` | 38 | +0.08363 | -0.01901 | +0.10264 |

The raw high percentages are therefore not all equivalent. `l`, `r`, `kaiin`, `aiin`, `eedy`, and small-sample `cheey` retain a positive local residual. `ain` is essentially null after full controls, and `ar` is negative. This is why selected 70–80% cells cannot by themselves establish a rule: the stem-specific base rate and manuscript production state matter.

## Section / Currier / hand replication

| stratum | eligible n | gain bits/event |
|---|---:|---:|
| currier=A | 5724 | +0.01126 |
| currier=B | 10787 | +0.01928 |
| section=Balneological | 3136 | +0.03079 |
| section=Herbal-A | 3929 | +0.01028 |
| section=Pharmaceutical | 506 | +0.03824 |
| section=Stars | 4820 | +0.02988 |
| hand=S1 | 4082 | +0.01073 |
| hand=S2 | 4510 | +0.02529 |
| hand=S3 | 5061 | +0.02879 |
| hand=S4 | 414 | +0.02641 |

Positive effects occur independently in Currier A and B and in the major Davis-hand approximations S1–S3. This makes a single-section or single-hand pooling artefact unlikely.

## Binding interpretation

**SUPPORTED: boundary-local contextual head selection as a surface-production effect.** Once the next token remainder is fixed, the immediately preceding final contains small but reproducible additional information about which first character is realised. The aggregate effect is modest (~0.0155 bits/event under the frozen τ=20 model), but it is 8.68 matched-null SD, positive in all five folio folds, and drops strongly at lag 2/3.

This is stronger than simply confirming edge mutual information. It identifies a more specific mechanism-compatible statement: an abstract next-token family can have its realised head modulated by the immediately preceding boundary environment.

It does **not** establish phonetic vowels, plaintext letters, semantic morphology, or a cipher. It is compatible with a contextual surface realiser / graphotactic production mechanism and provides a plausible source for part of the ~0.20-bit cross-boundary coupling.

## Execution note

The frozen plan requested 100 within-line permutations. Repeated full runs hit the execution ceiling after 25–30 permutations. The final reported null uses the first 25 completed deterministic permutations. No model feature, eligibility rule, fold definition, smoothing parameter, decision criterion, or observed target statistic was changed.