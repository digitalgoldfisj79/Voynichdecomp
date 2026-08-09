# BnF 7342 free-switch M19 v0.7 — qualification stop

Date: 2026-08-09
Protocol freeze: `7dc86dd68eace84c47654f9c8061ecd19be6d039`
Initial runner: `0ccea68e5eef0b551cff7cb2703c20c9868e294c`
Amendment 001: `e0eb4b5260e1aa538ea9ebd43171de6c4022b6f1`
Amended runner: `afb3202a80369e429ab57b9fd53850c41521562a`
HF jobs: initial `6a78195bda2af92a634efe44` errored before scoring; amended `6a7819963e1f34a7e32bfddb` cancelled after decisive gate failure.

## Verdict

**v0.7 LANGUAGE-RANKING INSTRUMENT NOT QUALIFIED. NO VOYNICH INFERENCE.**

The first scored known-plaintext control was Latin replicate 0. Under the target Latin model the fitted surface-glyph→numerical-value mapping achieved **1.0000 weighted held-out mapping accuracy**: the hidden 19-value numerical assignment was recovered exactly at the occurrence-weighted level.

However the frozen primary ranking statistic — held-out mapping-permutation z — did not identify the plaintext language:

| language | permutation z |
|---|---:|
| French | 4.2705 |
| German | 3.9986 |
| Spanish | 3.7908 |
| Latin | **3.7375** |
| Italian | 3.7137 |
| Greek | 3.2323 |
| Arabic | 2.9901 |
| Hebrew | 2.9651 |

Latin ranked 4th, so the frozen requirement of 8/8 correct controls became impossible after the first scored control. The run was cancelled to avoid unnecessary compute and never entered the Voynich stage.

## Important diagnostic

Unlike M57 v0.5, this is **not primarily a mapping-recovery failure**. The numerical channel itself was recovered perfectly on the first control. What fails is using a pairwise mapping-permutation z as a cross-language classifier after the numerical stream is recovered.

This justifies instrument development on controls only: retain the M19 mapping model, but evaluate recovered numerical streams with an exact/forward hidden-letter likelihood under the known BnF emission channel before any further Voynich test is permitted.
