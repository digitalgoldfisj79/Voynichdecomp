# VBM Joachim Exact v9 — Q2 closeout

Date: 2026-09-01
Branch: `experiment/vbm-joachim-exact-v9-20260901`
Protocol: `VBM_JOACHIM_EXACT_V9_Q2_SYNTH_IDENT_PROTOCOL.md`
Formal CAL Hugging Face job: `6a97355c21c5aa7c8364c4d8`
Job status: COMPLETED

## Decision

`GLOBAL_CODEBOOK_NOT_IDENTIFIABLE_WITH_CURRENT_SOLVER`

Formal CAL failed. Under the frozen protocol this closes Q2 at CAL. No VAL run is permitted and no Voynich TRAIN/HOLDOUT or H1/C1 language fit is opened from this branch.

## Positive-control recovery

The decisive failure is not merely weak discrimination against adversaries. The blind solver fails to recover even synthetic data generated exactly by the stipulated reusable VBM architecture.

### DE_GLOBAL

| rep | LANG_OK | REC_B | REC_N | REC_CHAR | HOLD_ADV | STAB |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | true | 0.4521 | 0.0311 | 0.0875 | 1.5434 | 0.6538 |
| 1 | true | 0.3973 | 0.0347 | 0.0715 | 1.4771 | 0.6743 |
| 2 | true | 0.5674 | 0.0603 | 0.0835 | 1.4235 | 0.6222 |

### IT_GLOBAL

| rep | LANG_OK | selected | REC_B | REC_N | REC_CHAR | HOLD_ADV | STAB |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 0 | false | DE | 0.2558 | 0.0000 | 0.0438 | 1.5146 | 0.7318 |
| 1 | false | DE | 0.2271 | 0.0000 | 0.0436 | 1.4562 | 0.6722 |
| 2 | false | DE | 0.2561 | 0.0000 | 0.0550 | 1.4068 | 0.8020 |

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
- IT_FRESHLINE: 1.516594
- MARKOV1: 1.390864
- SHUFFLED_GLOBAL: 1.483215

The minimum positive family median is 1.456187 while the maximum negative/adversarial median is 1.516594. Therefore `sep_ADV=false`.

Family median `STAB`:

- DE_GLOBAL: 0.653802
- IT_GLOBAL: 0.731762
- DE_FRESHLINE: 0.487898
- IT_FRESHLINE: 0.608541
- MARKOV1: 0.696377
- SHUFFLED_GLOBAL: 0.659520

The minimum positive family median is 0.653802 while the maximum negative/adversarial median is 0.696377. Therefore `sep_STAB=false`.

Thus the two statistics intended to distinguish a genuinely reusable global codebook from fresh-line and structured non-language controls are both nonseparable in formal CAL.

## Relation to Q1

Q1 had already shown that fresh one-line VBM fits are non-evidential: the fixture itself requires about 122.93 fresh key bits for 30 plaintext characters (4.10 key bits per plaintext character), and the real Voynich topology was not more selective than the structural-shuffle null (`z = -1.342`). Its frozen decision was `FRESH_FIT_NO_EVIDENTIAL_WEIGHT_REQUIRE_GLOBAL_CODEBOOK`.

Q2 tests that required escape route. The current blind reusable-codebook solver does not recover its own synthetic source model and does not separate positives from binding adversaries. Consequently no claim about Voynich plaintext, language, or a VBM codebook is licensed by v9.

## What this does and does not establish

This is a failure of **identifiability under the frozen v9 representation, language models, codebook size, and solver**. It is not a mathematical proof that every possible vowel-bridge cipher is impossible. It does establish that Joachim's illustrative line-level feasibility demonstration cannot be upgraded into evidence by the current global-codebook route, because the proposed inference machinery cannot first recover known synthetic instances of that architecture.

Any future VBM programme must begin with a new preregistered synthetic-identifiability protocol and demonstrate recovery there before touching Voynich data. It may not tune a solver on Voynich output and then cite that same fit as evidence.
