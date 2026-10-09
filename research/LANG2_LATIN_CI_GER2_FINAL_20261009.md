# LANG2 / Latin *Circa instans*: faithful GER2 replication, final adjudication

Date: 2026-10-09. Outcome: **FAIL frozen promotion gate; weak directional but unresolved**. All eight independent terminal `VGER2_RESULT_JSON` records reported `status=complete` with exact same VMS fold SHA `e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888`.

## Scientific firewall
Original GER2 solver pinned `e36031a9a6b8b67fcebb4d6f4af1c3753fad4287`, exact original runtime `98b7324fc65aa9b14242920f6931a8d1256064df`, replication wrapper `10e8d498c745f833223b58ac2fdbfb66c7602371`. Zero scientific settings changed. Full 10 rounds × 150 epochs, original E step and checkpoint selection, discovery folds 2/3, validation fold4, physically sealed strict-final folds0/1, first two line tokens excluded, novel vocabulary 735 each, known target 4103 / 10000, MCF demand221. True 10k Latin vocabulary vs unique within-form anagrams; all 10k null vocabulary entries unique (5 unavoidable unchanged). Source: a **single historical Latin witness**, `Circa instans`, parsed pickle SHA `377daa2a9c2403e6b8e10146a9c67f9e6b8144e959f232ec9096cdd0a4ae81f1`; source generalization has not been qualified.

## Four matched pairs

| Model seed | Null scramble seed | Validation real | Validation null | Strict final real | Strict final null | Validation win | Strict win |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1234 | 9001 | 0.056707170 | 0.083930790 | 0.041440330 | 0.091110371 | YES | YES |
| 2026 | 7001 | 0.033427071 | 0.158083528 | 0.046014737 | 0.152395606 | YES | YES |
| 17 | 11003 | 0.023723979 | 0.042749293 | 0.022844600 | 0.024628948 | YES | YES |
| 73 | 17011 | 0.086243689 | 0.076535217 | 0.091014162 | 0.093651697 | NO | YES |

## Pre-registered effect calculations

- **validation:** real mean 0.050025477, null mean 0.090324707, benefit (null−real) 0.040299230, **null sample SD 0.048599417**, effect/SD 0.82921, paired wins **3/4**. Per-seed matched effects [0.027223621, 0.124656457, 0.019025315, -0.009708472].
- **strict_final:** real mean 0.050328457, null mean 0.090446656, benefit (null−real) 0.040118198, **null sample SD 0.052218630**, effect/SD 0.76827, paired wins **4/4**. Per-seed matched effects [0.049670041, 0.106380869, 0.001784349, 0.002637535].

Frozen gate: (1) real mean lower on both; (2) strict-final effect/SD >2; (3) validation effect/SD >2; (4) ≥3 of 4 matched wins on each; (5) sign transfer fold4 to sealed folds0/1. **Mean direction passes; 3/4 validation and 4/4 final seed-count gates pass; both >2 SD requirements FAIL. No inferential promotion, no decrypted mapping inspection.**

Relative to GER2 German arms (also failed): Bavarian val z0.8458, final z0.9904; Alemannic val z1.5790, final z2.4819; Latin CI val z0.8292, final z0.7683. These are *within-arm* standardized real-versus-anagram effects, **not posterior language ranks**. Never compare raw edit costs across different source vocabularies as source-language evidence.

## Immutable terminal job sources

- Seed 1234 real: https://huggingface.co/jobs/Digitalgoldfish79/6ac9236c095c578089307e16; null: https://huggingface.co/jobs/Digitalgoldfish79/6ac9237d095c578089307e1f
- Seed 2026 real: https://huggingface.co/jobs/Digitalgoldfish79/6ac92b0a095c578089308288; null: https://huggingface.co/jobs/Digitalgoldfish79/6ac92b17fee2c9007017b1c7
- Seed 17 real: https://huggingface.co/jobs/Digitalgoldfish79/6ac92b12fee2c9007017b1c5; null: https://huggingface.co/jobs/Digitalgoldfish79/6ac92b18095c578089308292
- Seed 73 real: https://huggingface.co/jobs/Digitalgoldfish79/6ac92b14095c57808930828c; null: https://huggingface.co/jobs/Digitalgoldfish79/6ac92b1afee2c9007017b1d1

Original batched seed packs `6ac9239dfee2c9007017accb` (real) and `6ac923b2fee2c9007017acdf` (null) suffered **1800-second infrastructure timeout**, no results adjudicated from them. One-seed reruns used timeout3600 and were full original GER2, not shortened models.

## Scientific interpretation and next action

The result is positive *directionally* but noisy. Even with every strict-final seed pair favorable, the 0.040118 cost advantage is only 0.768 null SD. It **neither validates Latin as upstream language nor excludes it**. German was not validated either. One witness and four noisy training/null seeds limit sensitivity. Before any follow-up language adjudication, independently qualify source-corpus adequacy and instrument power; use multiple independent manuscript witnesses and faithful matched nulls rather than unqualified raw-score rankings or post-hoc thresholds. LANG2-B multilingual common renderer remains a separate unqualified hypothesis and main VMS R4 recovery ladder must remain untouched.

Reference frozen preregistration: https://github.com/digitalgoldfisj79/Voynichdecomp/blob/961dbbe1d53d6c30a26432be99f4d72ee270b7ab/research/LANG2_STRICT_GER2_PREREG_20261009.md
