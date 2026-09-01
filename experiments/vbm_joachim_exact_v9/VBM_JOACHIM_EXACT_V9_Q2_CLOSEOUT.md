# VBM Joachim Exact v9 — Q2 closeout

Date: 2026-09-01
Branch: `experiment/vbm-joachim-exact-v9-20260901`
Protocol: `VBM_JOACHIM_EXACT_V9_Q2_SYNTH_IDENT_PROTOCOL.md`
Pre-output implementation patch: `vbm_joachim_exact_v9_q2_synth_ident_patch1.py` (STAB weighting corrected to frozen HOLDOUT definition before smoke output)
Smoke Hugging Face job: `6a97354c0718b0f6d890e019`
Formal CAL Hugging Face job: `6a9735850718b0f6d890e01f`
Post-closeout oracle diagnostic job: `6a97379b21c5aa7c8364c530`

## Decision

`GLOBAL_CODEBOOK_NOT_IDENTIFIABLE_WITH_CURRENT_SOLVER`

Formal CAL failed. Under the frozen protocol this closes Q2 at CAL. No VAL run is permitted and no Voynich TRAIN/HOLDOUT or H1/C1 language fit is opened from this branch.

## Positive-control recovery

The decisive failure is not merely weak discrimination against adversaries. The blind solver fails to recover even synthetic data generated exactly by the stipulated reusable VBM architecture.

### DE_GLOBAL

| rep | LANG_OK | REC_B | REC_N | REC_CHAR | HOLD_ADV | STAB |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | true | 0.4521 | 0.0311 | 0.0875 | 1.5434 | 0.6484 |
| 1 | true | 0.3973 | 0.0347 | 0.0715 | 1.4771 | 0.6598 |
| 2 | true | 0.5674 | 0.0603 | 0.0835 | 1.4235 | 0.6245 |

### IT_GLOBAL

| rep | LANG_OK | selected | REC_B | REC_N | REC_CHAR | HOLD_ADV | STAB |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 0 | false | DE | 0.2558 | 0.0000 | 0.0438 | 1.5146 | 0.7295 |
| 1 | false | DE | 0.2271 | 0.0000 | 0.0436 | 1.4562 | 0.6863 |
| 2 | false | DE | 0.2561 | 0.0000 | 0.0550 | 1.4068 | 0.7970 |

Frozen positive gates required at least 5/6 global-positive replicates to satisfy each of:

- `LANG_OK=true`: observed **3/6**;
- `REC_CHAR >= 0.50`: observed **0/6**;
- `REC_B >= 0.70` and `REC_N >= 0.40`: observed **0/6**.

All three recovery gates fail, two of them maximally.

## Adversarial discrimination

Family median `HOLD_ADV`:

- DE_GLOBAL: 1.477113
- IT_GLOBAL: 1.456187
- DE_FRESHLINE: 1.436660
- IT_FRESHLINE: **1.516594**
- MARKOV1: 1.390864
- SHUFFLED_GLOBAL: **1.483215**

The minimum positive family median is 1.456187 while the maximum negative/adversarial median is 1.516594. Therefore `sep_ADV=false`.

Family median `STAB` using the frozen HOLDOUT weighting:

- DE_GLOBAL: 0.648432
- IT_GLOBAL: 0.729462
- DE_FRESHLINE: 0.481659
- IT_FRESHLINE: 0.584940
- MARKOV1: **0.657484**
- SHUFFLED_GLOBAL: **0.674177**

The minimum positive family median is 0.648432 while the maximum negative/adversarial median is 0.674177. Therefore `sep_STAB=false`.

Thus the two statistics intended to distinguish a genuinely reusable global codebook from fresh-line and structured non-language controls are both nonseparable in formal CAL.

## Q2D oracle diagnosis

Q2D was frozen and run only after Q2 had already failed. It cannot promote the model or reopen target access. Its purpose was to distinguish a merely bad optimiser from a non-identifying objective.

The known true synthetic key contains real signal:

- in **6/6** global positives the true key scored above every one of 200 multiplicity-preserving random keys;
- median `ORACLE_ADV` over the random-key median was **1.44678 nats/character**;
- the generating-language LM beat the other LM on the known plaintext in **6/6** cases, so the Q2 Italian-to-German failure is not explained by a trivial raw-LM calibration reversal.

However, the true key is not a coordinate optimum of the language objective in any replicate. Starting exactly at truth and changing only one dictionary entry at a time, the fraction of occurring entries with at least one wrong value that improves the FIT+SELECT LM objective was:

- DE rep 0: 0.3000
- DE rep 1: 0.3372
- DE rep 2: 0.3293
- IT rep 0: 0.2165
- IT rep 1: 0.3263
- IT rep 2: 0.2414

Median: **0.31316**. The truth was a coordinate local optimum in **0/6** replicates.

This is the key diagnostic result. The search problem does contain a truth signal relative to random mappings, but the finite-sample language-likelihood surface does not uniquely prefer the generating dictionary. It actively offers many locally improving moves away from truth. That makes 'best-looking language output' non-identifying: a wrong key can be rewarded even when a correct global key is known to exist.

## Relation to Q1

Q1 had already shown that fresh one-line VBM fits are non-evidential: the fixture itself requires about 122.93 fresh key bits for 30 plaintext characters (4.10 key bits per plaintext character), and the real Voynich topology was not more selective than the structural-shuffle null (`z = -1.342`). Its frozen decision was `FRESH_FIT_NO_EVIDENTIAL_WEIGHT_REQUIRE_GLOBAL_CODEBOOK`.

Q2 tests that required escape route. The current blind reusable-codebook solver does not recover its own synthetic source model and does not separate positives from binding adversaries. Q2D further shows that this is not simply a matter of adding more random restarts: the scoring objective itself rewards many deviations from the true key.

## What this does and does not establish

This is a failure of **identifiability under the frozen v9 representation, language models, codebook size, and inference objective**. It is not a mathematical proof that every possible vowel-bridge cipher is impossible.

It does establish that Joachim's illustrative line-level feasibility demonstration cannot presently be upgraded into evidence by either of the tested routes:

1. fresh line-specific readings have too much dictionary freedom and fail the Q1 specificity/MDL audit;
2. a reusable global codebook cannot be identified by the Q2 language-optimisation route even on synthetic truth, and its observable fit/stability statistics overlap binding adversaries.

No claim about Voynich plaintext, language, or a VBM codebook is licensed by v9. Any future VBM programme must begin with a genuinely new synthetic-identifiability instrument whose objective recovers known keys before touching Voynich data. It may not tune a decoder on Voynich output and then cite that same linguistic fit as evidence.
