# VBM Bavarian–German conditional discriminator v5 — preregistration

Date: 2026-08-11
Branch: `experiment/vbm-bg-conditional-v5-20260811`
Namespace: `VBMBGCONDV5`

## Question

Can the VBM representation support a **prospectively validated conditional Bavarian-vs-German discrimination** that is independent of the v4 HMM likelihood, and only then pass a separately qualified language-vs-flexible-null gate?

This is not a fresh test of the already-opened VBM H1 topology: v1 previously established an H1 Bavarian CV-topology signal. v5 is a formal successor intended to determine whether that signal survives a stronger cross-corpus discriminator and an explicit non-language null gate. `VBM_C1` remains the genuinely sealed confirmation set.

## Frozen target integrity

- `VBM_H1 = f28v f31v f88r f5r f34r f81v`.
- `VBM_C1 = f85r1 f53v f33r f10r f23r f111r`.
- Q0 and Q1 may not read or score either target set.
- H1 may be scored only if Q0 and Q1 both qualify.
- C1 may be opened only if H1 passes every registered target gate.
- No candidate plaintext may be inspected before C1 disposition.

## Stage Q0 — independent Bavarian/German topology discriminator

### Representation

Only the binary consonant/vowel topology is used. No HMM emissions, decoded letters, Voynich token identities, word identities, or target-trained parameters enter the classifier.

The normalized 19-letter alphabet and vowel set are inherited unchanged from VBM v1.

### Discovery corpora

- Bavarian: `bavarian-nlp/barwiki-20250720`.
  - deterministic article split inherited from v1: IDs mod 10 <6 provide generative-model training; IDs mod 10 >=6 are controls.
- German: UD German GSD.
  - train file provides generative-model training;
  - dev+test provide controls.

The control streams are deterministically split by sentence hash into classifier-fit and discovery-validation halves. No target data are involved.

### Independent transfer corpora

- Bavarian: UD Bavarian MaiBaam `bar_maibaam-ud-test.conllu`, excluding metadata genres `wiki` and `social`. This preserves non-Wikipedia fiction, grammar-example and non-fiction material and spans multiple Bavarian dialect groups.
- German: UD German PUD `de_pud-ud-test.conllu`.

No parameter is refit on the transfer corpora.

### Feature family

Eight fixed topology features:

1–6. per-event Bavarian-minus-German log-likelihood ratios from binary CV Markov models of orders 1 through 6;
7. consonant-run-length log-likelihood ratio, run lengths capped at 8;
8. vowel-run-length log-likelihood ratio, run lengths capped at 8.

The generative feature models are fit only on the discovery training corpora. Feature standardisation and a ridge logistic classifier (`lambda = 1.0`) are fit only on classifier-fit control windows. No order, feature, regularisation or sign is selected from target behaviour.

Nominal window length is 1,800 CV events for discovery and 1,200 for independent transfer; features are length-normalised. Windows are deterministic, non-overlapping chunks after a namespace-derived offset. At least 8 windows per class are required for each binding evaluation.

### Q0 qualification

Discovery validation must satisfy all:
- balanced accuracy >= 0.90;
- Bavarian recall >= 0.85;
- German recall >= 0.85.

Independent transfer must satisfy all:
- balanced accuracy >= 0.80;
- Bavarian recall >= 0.75;
- German recall >= 0.75;
- at least 8 evaluated windows per class.

If Q0 fails, stop. No HMM calibration and no target access.

If Q0 passes, freeze from the transfer Bavarian windows a target-strength floor `TAU_BG = max(0, p10(logit_Bavarian_transfer_correct))`. This threshold is frozen before any target access.

## Stage Q1 — language-vs-flexible-null detector

Run only after Q0 passes.

The v4 typed soft HMM and registered flexible null family are retained, but exact-language ranking is not used as the primary criterion. The binary statistic is:

`DELTA_LANG = max(HMM_Bavarian, HMM_German) - best_registered_nonlanguage_null`.

Only source-grounded Amadi `CYCLE` bridge scheduling is admitted in v5. This is a narrower model than v4/v4.1, not a retrospective rescue of those closed calibrations.

### Positive controls

Two entirely fresh namespaces, calibration and validation. In each namespace:
- Bavarian and German;
- all six frozen core-homophone allocation regimes;
- CYCLE bridge schedule only.

Thus 12 fresh positives per namespace.

### Negative controls

24 fresh negatives per namespace, six seeds from each of four destruction families:
1. typed IID surface generation matched to a positive control's type fraction and within-type symbol marginals;
2. typed first-order Markov generation preserving sequence-length distribution but not language higher-order structure;
3. periodic typed slot generation with period deterministically chosen from 2–8;
4. within-sequence block shuffle using fixed block size 3, preserving local surface fragments while destroying longer ordering.

No negative is derived from Voynich.

### Threshold calibration and validation

On the calibration namespace choose the **largest** threshold `TAU_LANG` among observed DELTA values that satisfies calibration FPR <=0.05 while maximising TPR; ties choose the larger threshold.

On the untouched validation namespace require:
- TPR >=0.85;
- FPR <=0.05;
- balanced accuracy >=0.90;
- both Bavarian and German positive-control recall >=0.80.

If Q1 fails, stop before target.

## Stage Q2 — H1 application

Run only if Q0 and Q1 pass.

For aggregate H1 require both:
1. `DELTA_LANG >= TAU_LANG`;
2. frozen Q0 topology classifier logit `>= TAU_BG` and Bavarian direction.

Cross-folio topology requirement:
- at least 5/6 H1 folios have Bavarian-direction logit >0;
- median folio logit >0.

No plaintext is inspected. If any H1 gate fails, stop with C1 sealed.

## Stage Q3 — sealed C1 confirmation

Run once only if every H1 gate passes.

Apply the identical frozen Q1/Q0 instruments and thresholds. Require:
1. aggregate `DELTA_LANG >= TAU_LANG`;
2. aggregate Bavarian topology logit `>= TAU_BG`;
3. at least 5/6 C1 folios Bavarian-direction logit >0;
4. median C1 folio logit >0.

Only a Q3 pass permits the statement: **prospectively confirmed Bavarian-compatible VBM signal under v5**. It still does not establish a decipherment or readable Bavarian plaintext.

## Stop rules

- No threshold relaxation after any target statistic is generated.
- No feature additions/deletions after Q0 begins.
- No extra HMM starts, alternative homophone families or corpus substitutions after Q1 begins.
- If a qualification gate fails, close v5 rather than tuning through the target.
- Paid compute jobs must be cancelled immediately when the next binding gate becomes mathematically impossible, and running-job status must be checked at closeout.
