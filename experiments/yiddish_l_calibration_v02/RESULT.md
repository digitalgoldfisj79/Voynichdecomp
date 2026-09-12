# Yiddish–German L calibration transfer v0.2 — result

## RETRACTIONS / CONTROLLING LIMITS

- v0.1 remains permanently quarantined for the ReF token-unit defect.
- v0.1b remains `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`; this result does not rewrite it.
- The post-v0.1b same-map paired-null 10/10 diagnostic is non-controlling and was not used here.
- This is an L-unseen transfer panel, not fresh whole-programme confirmation. Voynich was never loaded.

**Status: `L_RECOVERY_V02_TRANSFER_NOT_RESOLVED`**

## Frozen-gate outcome

- correct_language_recovery: **FAIL**
- wrong_language_selectivity: **PASS**
- deployable_accuracy: **FAIL**
- matched_label_null_2sd: **PASS**
- beats_unigram_by_2_errors: **PASS**
- beats_shuffle_by_2_errors: **PASS**
- complete_non_degenerate: **PASS**
- leave_one_family_out_stable: **FAIL**

## Primary headline

Primary family accuracy = 0.688; effect over exact label null = +0.344, null SD = 0.099150, effect/nullSD = 3.466968062972152; exact p = 0.000078.
Unigram nuisance accuracy = 0.500; within-word-shuffle nuisance accuracy = 0.375.

## Family results

| truth | family | median Z_ind | call | correct M0 passes | wrong-model M0 passes | unigram call | shuffle call |
|---|---|---:|---|---:|---:|---|---|
| german | F004 | -1.0563 | german | 32/32 | 0/32 | yiddish | abstain |
| german | F028 | -0.6498 | abstain | 0/32 | 0/32 | yiddish | abstain |
| german | F057 | -0.5028 | abstain | 32/32 | 0/32 | yiddish | abstain |
| german | F128 | -0.8748 | abstain | 32/32 | 0/32 | yiddish | abstain |
| german | F166 | +0.1014 | abstain | 0/32 | 0/32 | yiddish | abstain |
| german | F249 | -0.9228 | abstain | 32/32 | 0/32 | yiddish | abstain |
| german | F300 | -1.0599 | german | 32/32 | 0/32 | yiddish | abstain |
| german | F313 | -1.4016 | german | 32/32 | 0/32 | yiddish | abstain |
| yiddish | ashkenaz_un_polak_1675 | +2.3889 | yiddish | 32/32 | 0/32 | yiddish | abstain |
| yiddish | glikl_1705 | +1.9114 | yiddish | 32/32 | 0/32 | yiddish | yiddish |
| yiddish | magen_1624 | +2.1428 | yiddish | 32/32 | 0/32 | yiddish | yiddish |
| yiddish | magid_1600 | +1.9999 | yiddish | 32/32 | 0/32 | yiddish | abstain |
| yiddish | messiah_1666 | +2.0893 | yiddish | 32/32 | 0/32 | yiddish | yiddish |
| yiddish | moses_1750 | +1.9549 | yiddish | 32/32 | 0/32 | yiddish | yiddish |
| yiddish | purim_1697 | +2.0180 | yiddish | 0/32 | 0/32 | yiddish | yiddish |
| yiddish | vilna_1692 | +2.3766 | yiddish | 32/32 | 0/32 | yiddish | yiddish |

## Post-outcome bounded diagnostic

This diagnostic does not alter the frozen result.

- `F028`: correct German direction on 32/32 keys, but M0 recovery fails; mean correct-model atom recovery ≈0.836 and median whole-word recovery ≈0.406.
- `F166`: genuine structural failure; mean correct-model atom recovery ≈0.834, median whole-word recovery ≈0.564, and only 5/32 key-level contrasts have the correct German sign.
- `purim_1697`: strong Yiddish direction on 32/32 keys, but whole-word recovery is ≈0.760, below the frozen 0.800 mechanism gate despite atom recovery ≈0.932.

## Interpretation boundary

A full transfer pass still requires genuinely fresh external historical-Yiddish confirmation before L can be issued. This transfer did not pass. L therefore remains unresolved and no result here licenses target/Voynich scoring. C7 remains sealed.
