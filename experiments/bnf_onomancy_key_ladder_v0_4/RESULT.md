# BnF 7342 key-granularity ladder v0.4 — result

Date: 2026-08-09
Branch: `experiment/bnf-onomancy-key-ladder-v0.4-20260809`
Protocol freeze: `3db3f8be97aa09764001cf9ee249d2a171a38ddb`
Initial runner: `8a221c63f0bb4dc8417235226a9fe8b1114a7869`
Amendment 001: `21f1df62fd1595b97d21ac262771b15c2425bd67`
Amendment 002: `57ed399a508f84548bb493dbcd9843a4f51e3bbc`
Amendment 003: `7f1408d5366f57b286cd01e50943bd8226464e6e`
Final wrapper: `a4968ffbc16e65641e8279ba5f5448b92cd45c16`

## Verdict

**NO RESOLVING T2 PIECEWISE-KEY SIGNAL.**

The bounded ladder tested increasingly flexible coarse key reuse for the conservative BnF-derived four-homophone T2 adaptation. K-CURRIER, K-SECTION and K-SECTION×CURRIER all passed their synthetic positive-control gates but failed the preregistered language-separation criterion on Voynich. K-QUIRE failed its positive-control gate and therefore was not applied to Voynich.

This does not reject other mechanism classes. In particular it does not test the manuscript's literal single-table letter→number mappings as many-to-one numerical encodings.

## K-CURRIER

HF job: `6a780e6bda2af92a634efdbf` — completed.

- 3 keys: Currier A/B/C.
- held-out mapped glyph coverage: 42,823/42,828 = 99.9883%.
- synthetic gate: PASS — 4/4 languages correct; median plaintext accuracy 0.9940; minimum 0.9016; median mapping z 21.6315.

Voynich held-out mapping z:

| language | z | lexical z |
|---|---:|---:|
| Spanish | 10.6448 | 6.6048 |
| Arabic | 9.3076 | 11.1869 |
| Greek | 8.9848 | 9.1748 |
| Latin | 8.3335 | 5.1568 |
| Italian | 8.1755 | 9.3322 |
| Hebrew | 7.9964 | 10.5702 |
| French | 7.4474 | 9.4955 |
| German | 7.3144 | 10.7165 |

Top-v-second margin = 1.3372, below frozen 5. Verdict: **NO SIGNAL**.

## K-SECTION

HF job: `6a780fc03e1f34a7e32bfcf7` — completed.

- 10 content-section keys.
- all ten frozen sections evaluable.
- held-out mapped glyph coverage 38,568/38,576 = 99.9793%.
- synthetic gate PASS: 4/4 correct; median accuracy 0.9941; min 0.8898; min group-accuracy coverage 0.8772; median z 35.6713.

Voynich:

| language | z | lexical z | MDL gain-minus-key-penalty nats |
|---|---:|---:|---:|
| Spanish | 14.6798 | 10.9415 | +5,979 |
| Arabic | 14.5722 | 13.6729 | -1,897 |
| Hebrew | 13.8319 | 10.7841 | -1,118 |
| Greek | 13.5438 | 12.7775 | +469 |
| Italian | 12.6919 | 7.0384 | +4,993 |
| Latin | 12.3818 | 7.8816 | +6,337 |
| French | 10.8314 | 12.0097 | +10,337 |
| German | 10.6767 | 9.8670 | -10,958 |

Top-v-second margin = 0.1076. The fact that every language crosses z=10 demonstrates flexibility, not eight simultaneous plaintexts. Verdict: **NO SIGNAL**.

## K-SECTION×CURRIER

HF job: `6a7811283e1f34a7e32bfd10` — completed.

- 13 evaluable keys.
- two tiny cells excluded prospectively: Cosmological|B and Herbal-A|C.
- evaluable character coverage 99.6706%.
- held-out mapped glyph coverage 36,784/36,802 = 99.9511%.
- synthetic gate PASS: 4/4 correct; median accuracy 0.95995; min 0.89256; min group-accuracy coverage 0.87919; median z 38.4803.

Voynich:

| language | z | lexical z | MDL gain-minus-key-penalty nats |
|---|---:|---:|---:|
| Arabic | 15.8453 | 19.1998 | +17,187 |
| Hebrew | 14.5890 | 15.2740 | +13,969 |
| Spanish | 14.5869 | 17.2950 | -1,844 |
| Italian | 14.4625 | 11.1420 | -484 |
| Latin | 13.9106 | 11.5931 | -2,241 |
| Greek | 12.9778 | 15.2089 | +5,377 |
| German | 12.8153 | 10.2685 | -2,417 |
| French | 12.7159 | 11.1105 | -5,643 |

Top-v-second margin = 1.2563. Again the entire panel is elevated. Verdict: **NO SIGNAL**.

## K-QUIRE

Initial job `6a7812b0da2af92a634efdde` stopped in synthetic controls due a train-unseen generated homophone; Amendment 003 prospectively harmonized synthetic-control handling with Amendment 001.

Final job: `6a7813c83e1f34a7e32bfd36`.

- 16 evaluable quire keys; Q10 and Q12 excluded by frozen minimum-data rule.
- evaluable corpus coverage 97.3528%.
- Voynich held-out mapped coverage 39,397/39,415 = 99.9543%.
- synthetic heldout mapped-symbol coverage: Latin 99.9848%, Italian 99.9264%, German 99.9873%, Hebrew 100%.
- synthetic languages all rank correctly.
- median plaintext accuracy 0.9466; minimum 0.7733; median mapping z 39.3691.
- **P4 FAIL:** Hebrew has only 72.46% of held-out characters in groups with per-group character accuracy >=0.70, below frozen 80% requirement.

Therefore K-QUIRE verdict = **UNDERPOWERED: POSITIVE CONTROL FAIL**. No quire-level Voynich language score was generated.

## Methodological note

v0.4's implementation used a deterministic representative whole-page training sample targeting roughly 35k glyphs across groups; this sampling rule was not explicit in the original prose freeze and was recorded prospectively in Amendment 002. Consequently v0.4 is treated as a screening ladder. No positive result occurred, so no confirmatory v0.5 rerun was triggered.

## Interpretation

Increasing the number of T2 keys raises held-out language-model and dictionary-enrichment scores for essentially every candidate language. This is the signature of representational flexibility, not language identification. The preregistered between-language margin prevents those absolute scores from being misreported as Spanish, Arabic, Hebrew, etc.

The useful boundary is now clear: coarse fixed-key reuse from global → Currier → content section → section×Currier does not produce a unique language signal, while quire-level inference becomes insufficiently reliable on known controls. Further arbitrary key partitioning would not be evidentially useful.
