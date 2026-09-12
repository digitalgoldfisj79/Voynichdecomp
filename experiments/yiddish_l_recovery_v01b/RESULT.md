# Yiddish–German L recovery competition v0.1b — controlling development result

## RETRACTIONS / CONTROLLING LIMITS

1. Parent v0.1 run `34714545374` remains permanently inadmissible because it used the wrong ReF measurement unit. No v0.1 solver output was reused here.
2. The earlier claim that commit `0f9e30ee32185cc65591b2da4867fa3bb8b23574` itself contained the ReF XML tarball was false; it configured artifact packaging. The successful Candidate-2 corpus artifact came from run `34693265951`.
3. The six-leaf YidTakNL primary-script bridge remains failed. This experiment is secondary-normalized development only and cannot promote a primary-script Yiddish claim.
4. Voynich/C7 was never loaded and remains sealed.

**Controlling status: `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`**

Run: `34715591480`, head `6b387cab83df01b7057bc537fbc0e7fd892cb338`.

Result artifact: `yiddish-german-l-recovery-v01b-result`, artifact id `10304463654`, SHA-256 `6e0ee9c336e9dad5c76861856d80244b27328977bd4f6bcb2c60177aba11e6f5`.

Public-preflight artifact: artifact id `10304543359`, SHA-256 `9de2e155d135096828e1f3943e35a5def0f03adde14c697e252bc868e628e6b5`; emitted manifest SHA-256 `7249cfce246f2844812a684fec8b941d6155c7bf21ceba778b4c22b85c20b4cc`.

## Pre-outcome admissibility

The actual workflow and an independent read of its emitted public artifact established before outcome interpretation:

- all seven ReF XML SHA-256 values exactly equal the Candidate-2 freeze;
- corrected virtual-token word counts are exactly F014 19,097; F015 18,162; F016 8,124; F018 6,270; F034 5,663; F037 15,327; F148 20,282;
- BUILD↔DEVELOPMENT exact 8-word overlap = 0 in both languages;
- Yiddish 5-word overlaps are diagnostic only: `shir_1579↔sam_hayyim_1590` 1 type, `shir_1579↔lev_tov_1620` 2 types, `ester_1589↔bovo_1507` 2 types; German 5-word overlap = 0;
- no v0.1 solver output was reused;
- target/Voynich access flags are false;
- all ten fresh blind solver jobs completed before private truth was revealed.

## Frozen-gate result

- correct-language recovery: **PASS**
- wrong-language selectivity: **PASS**
- deployable family accuracy >=0.80 at frozen ±1.0 margin: **FAIL**
- exact matched-label-null >=2 SD: **PASS**
- beats unigram nuisance by >=2 family errors: **PASS**
- beats within-word-shuffle nuisance by >=2 family errors: **PASS**
- complete/non-degenerate: **PASS**
- leave-one-family-out frozen conclusion: **FAIL**

Primary family accuracy = **0.700**. Effect over the exact balanced-label null = **+0.350**, null SD = **0.130171**, effect/null-SD = **2.6888**, exact one-sided p = **0.003968**.

N1 unigram accuracy = **0.000**. N2 within-word-shuffle accuracy = **0.100**.

## Family-level outcome at the frozen ±1.0 rule

| truth | family | median Yiddish-minus-German z margin | call | correct M0 passes | wrong-model M0 passes |
|---|---|---:|---|---:|---:|
| German | F016 | -1.3770 | German | 32/32 | 0/32 |
| German | F018 | -0.9775 | abstain | 31/32 | 0/32 |
| German | F034 | -0.9941 | abstain | 32/32 | 0/32 |
| German | F037 | -0.9490 | abstain | 32/32 | 0/32 |
| German | F148 | -1.1447 | German | 31/32 | 0/32 |
| Yiddish | bovo_1507 | +1.4942 | Yiddish | 32/32 | 0/32 |
| Yiddish | cracow_letters_1588 | +2.0104 | Yiddish | 32/32 | 0/32 |
| Yiddish | kine_1648 | +2.3746 | Yiddish | 32/32 | 0/32 |
| Yiddish | lev_tov_1620 | +2.0240 | Yiddish | 32/32 | 0/32 |
| Yiddish | sam_hayyim_1590 | +1.3834 | Yiddish | 32/32 | 0/32 |

The three errors are all abstentions, not wrong-language calls.

## Diagnostic interpretation — not a repaired gate

The underlying recovery signal did **not** collapse:

- family-margin sign is correct for **10/10** relationship groups;
- key-level margin sign is correct in **306/320** cases: Yiddish 160/160, German 146/160;
- correct-language key recovery is essentially complete (Yiddish mean atom recovery 0.99993 and mean word recovery 0.99974; German 0.99645 and 0.98668), whereas wrong-language recovery is far lower (Yiddish-source wrong-model mean atom 0.26912 / word 0.01135; German-source wrong-model mean atom 0.41119 / word 0.02712);
- the failure is concentrated in the preregistered abstention margin: three German family medians lie just inside −1.0.

There is a clear calibration asymmetry: mean correct-model audit z is similar for Yiddish and German (8.43 vs 8.22), while the wrong-language model also scores German audit text more strongly (mean wrong-model z 7.14) than it scores Yiddish audit text (6.55). Thus the paired margin is narrower on German. This is a property to test, not a license to lower the threshold after seeing the result.

Frozen sensitivity remains diagnostic only:

- ±0.5: 10/10 family calls correct, accuracy 1.0; effect +0.500 over null, null SD 0.166667, effect/null-SD 3.0, exact p 0.003968;
- ±1.0: controlling accuracy 0.7;
- ±1.5: accuracy 0.3. The metric does not resolve this stricter sensitivity setting: effect +0.150 over null, null SD 0.076376, effect/null-SD 1.964.

No sensitivity result changes the frozen v0.1b verdict.

## Consequence

The defensible conclusion is narrower than either “Yiddish passed” or “Yiddish failed.” Under M0 on the secondary normalized representation, the correct language model is strongly and consistently recovery-selective relative to German, and the signal disappears under the registered nuisance controls. However, the preregistered target-usable abstention threshold is not calibrated robustly enough across the two languages, so **L is not qualified**.

Do not retune v0.1b. Any next experiment must be separately frozen and must address threshold/calibration asymmetry with genuinely fresh information. More importantly, because the primary-script bridge already failed, no normalized-representation L result by itself can license C7.
