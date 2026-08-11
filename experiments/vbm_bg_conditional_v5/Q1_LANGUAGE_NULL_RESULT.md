# VBM BG conditional v5 — Q1 language-vs-null result

Date: 2026-08-11
Namespace: `VBMBGCONDV5Q1`
HF job: `6a7b88f427caad61c6eac608`

## Binding result

**Q1 PASS.**

Frozen threshold:

`TAU_LANG = 0.21806702545882306`

Statistic:

`DELTA_LANG = max(HMM_Bavarian, HMM_German) - best_registered_nonlanguage_null`.

Only CYCLE bridge scheduling was used, as preregistered.

## Calibration namespace

12 fresh positives and 24 fresh negatives.

- TPR: **12/12 = 1.0000**.
- FPR: **0/24 = 0.0000**.
- balanced accuracy: **1.0000**.
- Bavarian recall: **6/6 = 1.0000**.
- German recall: **6/6 = 1.0000**.

Positive DELTA ranges:
- Bavarian: +0.218067 to +0.317064 nats/event.
- German: +0.318266 to +0.365984.

Hardest calibration negative: block-shuffle3 +0.159885.

## Untouched validation namespace

12 fresh positives and 24 fresh negatives, no threshold refitting.

- TPR: **12/12 = 1.0000**.
- FPR: **0/24 = 0.0000**.
- balanced accuracy: **1.0000**.
- Bavarian recall: **6/6 = 1.0000**.
- German recall: **6/6 = 1.0000**.

Validation positive DELTA values:

Bavarian:
- ANTI_SQRT +0.2228123667
- UNIFORM +0.2455741401
- SQRT_FREQ +0.2298721531
- FREQ_PROP +0.2783412243
- SUPER_FREQ +0.3098379719
- DIRICHLET_SKEW +0.3458460853

German:
- ANTI_SQRT +0.2651701536
- UNIFORM +0.2993627012
- SQRT_FREQ +0.3634127472
- FREQ_PROP +0.3581986834
- SUPER_FREQ +0.3428102161
- DIRICHLET_SKEW +0.3389854634

Validation negative families:
- typed IID: 0/6 false positives; median DELTA -0.034722.
- typed first-order Markov: 0/6; median +0.014879.
- typed periodic slot: 0/6; median -0.036710.
- block-shuffle3: 0/6; median +0.117023; maximum +0.129339.

## Consequence

The Q1 instrument is qualified prospectively. Q2 may now open `VBM_H1` using the already frozen `TAU_LANG = 0.21806702545882306` and Q0 `TAU_BG = 1.6272712366587183`.

No Voynich H1 or C1 data were loaded by Q1. No target HMM fit or plaintext was generated during Q1.
