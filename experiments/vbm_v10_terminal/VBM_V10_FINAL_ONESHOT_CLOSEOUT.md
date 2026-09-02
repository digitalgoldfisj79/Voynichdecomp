# VBM v10 — final one-shot closeout

Date: 2026-09-02
Job: `6a97afd50718b0f6d890fa84`
Hardware: 4× NVIDIA A100-SXM4-80GB
Status: **COMPLETED**

## Frozen verdict

`VBM_GLOBAL_KEY_NOT_RECOVERABLE_EVEN_COMPACT`

The compact global-codebook architecture failed every preregistered recovery gate at the maximum favourable Stage-A corpus size of 2,000 lines (1,600 FIT / 400 untouched HOLDOUT), with the correct language and exact 32-value nucleus inventory supplied to the solver.

No full-tail stress test was opened. No Voynich plaintext test was opened.

## Software qualification

The in-job GPU smoke passed:

- GPU/CPU exact-likelihood maximum absolute difference: `6.553705986789282e-07` (gate `<=1e-5`)
- coordinate-polish likelihood monotonicity: PASS
- deliberately corrupted known key improved in FIT likelihood
- map recovery moved slightly downward during likelihood polish (`0.92063 -> 0.91270`), a diagnostic sign that higher LM likelihood need not imply movement toward the true key.

## Binding O2 results

| Replicate | REC_B | REC_N | REC_CHAR | REC_B5 | REC_N5 | REC_CHAR5 | HOLD_LM | TRUE_KEY_HOLD_LM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DE-0 | 0.4508 | 0.0407 | 0.1951 | 0.4512 | 0.0408 | 0.1952 | -1.8498 | -1.8484 |
| DE-1 | 0.4458 | 0.0332 | 0.2661 | 0.4458 | 0.0333 | 0.2662 | -1.8195 | -1.8463 |
| DE-2 | 0.6745 | 0.0994 | 0.2914 | 0.6746 | 0.0995 | 0.2914 | -1.8139 | -1.8624 |
| IT-0 | 0.5061 | 0.0634 | 0.2944 | 0.5065 | 0.0633 | 0.2948 | -1.9586 | -1.8579 |
| IT-1 | 0.5012 | 0.0533 | 0.2839 | 0.5012 | 0.0534 | 0.2842 | -1.9389 | -1.8406 |
| IT-2 | 0.2432 | 0.0000 | 0.1055 | 0.2427 | 0.0000 | 0.1053 | -1.9322 | -1.8464 |

Coverage was essentially complete in every replicate (`COV_B ~= 1`, `COV_N ~= 0.999-1.000`), so failure is not attributable to unseen HOLDOUT surface types.

Frequent-only recovery is effectively identical to unrestricted recovery, so the failure is not a rare-type tail artifact.

## Frozen gate counts

At 2,000 lines:

- `REC_CHAR >= 0.80`: **0/6** (required >=5/6)
- `REC_B >= 0.70` and `REC_N >= 0.70`: **0/6** (required >=5/6)
- frequent-only `REC_CHAR5 >= 0.90`, `REC_B5 >= 0.80`, `REC_N5 >= 0.80`: **0/6** (required >=5/6)
- German language subgroup: 0/3 character pass, 0/3 key pass
- Italian language subgroup: 0/3 character pass, 0/3 key pass

Therefore the compact architecture fails the terminal criterion decisively.

## Critical interpretation

The most important result is stronger than a simple optimizer miss.

The fitted wrong keys produced held-out language-model scores close to the true-key scores. In two German replicates the recovered wrong key even scored better under the fixed held-out LM than the true generating key (`DE-1: -1.8195 vs -1.8463`; `DE-2: -1.8139 vs -1.8624`) while nucleus recovery remained only `3.3%` and `9.9%` respectively.

Across all six replicates, nucleus recovery was only `0-9.9%`, character recovery `10.6-29.4%`, yet held-out LM scores remained far above random-key baselines (`HOLD_ADV ~= 1.34-1.72`). This means many globally wrong dictionaries can generate text that is highly plausible to the supplied language model.

The in-job smoke reinforces this diagnosis: starting near the true key, exact likelihood improvement slightly reduced exact map recovery.

Thus the present VBM inverse problem is not merely difficult for the search instrument. Under the stated architecture and objective, language-model plausibility is substantially non-identifying: it does not reliably point back to the true codebook even when the language, value inventory, parser, and global-codebook assumption are all correct and the corpus is synthetic.

## Consequence for Joachim-style readings

This does **not** prove that no conceivable Vowel Bridge cipher could have generated Voynich.

It does establish that the present whole-nucleus VBM, with unrestricted global bridge/nucleus homophones of the form currently proposed, lacks a validated inverse inference procedure. Attractive plaintexts or high German/Italian language-model scores therefore cannot presently be treated as evidence that the recovered mapping is the generating mapping.

Any future rescue would need to add independently motivated constraints that make the inverse problem identifiable—for example a genuinely specified codebook prior, constrained homophone multiplicities, a constrained nucleus-length/value system, or another externally justified structural rule. Such a rescue would constitute a new, more restrictive cipher hypothesis and would require fresh preregistration and synthetic validation before touching Voynich.

## Programme stopping rule

Per the frozen protocol:

- Stage B full-tail stress test: **NOT OPENED**
- Voynich TRAIN/HOLDOUT global-codebook test: **NOT OPENED**
- further unconstrained plaintext-search claims: **NO EVIDENTIAL WEIGHT**

The current VBM programme closes here unless a materially more constrained and independently specified cipher model is proposed.
