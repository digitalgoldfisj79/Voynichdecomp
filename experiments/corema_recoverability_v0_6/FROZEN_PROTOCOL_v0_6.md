# Frozen protocol — CoReMA procedural recoverability v0.6

**Frozen:** 2026-07-25, before downloading or inspecting the formal CoReMA TEI/XML corpus in this phase.  
**Branch:** `experiment/voynich-corema-recoverability-v0.6-20260725`  
**Upstream status:** v0.4 remains `ABSTAIN_OOD`; v0.5 remains `CALIBRATION_FAILURE`.

## 1. Purpose

The v0.5 historical-music calibration could recognise notation families but failed known-field recovery. This phase supplies the missing real medieval procedural control: the University of Graz CoReMA corpus of cooking, medicinal, household and technical recipes.

The question is whether explicit operational roles can be recovered across unseen manuscripts from the surface sequence. Voynich is not inspected until the external gates are frozen and evaluated.

## 2. Source and grouping

Use every publicly accessible CoReMA annotated-detail object of the form:

`https://gams.uni-graz.at/o:corema.<manuscript>.recipes/TEI_SOURCE`

The manuscript is the indivisible cross-validation group. Nested subrecipes are separate sequences. Text units shorter than three alphanumeric tokens are excluded. Failed or unavailable endpoints are audited and not replaced.

CoReMA's semantic TEI supplies explicit elements including `ingredient`, `instruction`, `tool`, `time`, `date`, `dish`, `title`, and advisory/meta elements. The primary mutually exclusive token roles are assigned by frozen precedence:

1. `INGREDIENT`
2. `TOOL`
3. `TEMPORAL`
4. `OUTPUT`
5. `ACTION`
6. `META`
7. `OTHER`

A role enters the primary macro-F1 only when it has at least 100 tokens and occurs in at least three manuscripts. This support rule is fixed before acquisition.

## 3. Models

Five manuscript-grouped folds are used whenever at least five manuscripts are available.

### 3.1 Lexical recovery

Character 2–5-gram TF–IDF logistic regression on the current token and two-token context on each side. This establishes whether the semantic annotation can be recovered across manuscripts when surface identity is available.

### 3.2 Identity-neutral recovery

Three identity-neutral views are evaluated:

- manuscript-local frequency-rank context;
- within-token equality/repetition pattern context;
- numeric structure: token and neighbour lengths, recipe position, manuscript-local frequency/rank, repetition, character diversity and boundary indicators.

A random forest produces structural emissions. A first-order role model trained only on training manuscripts supplies HMM/Viterbi sequence smoothing.

### 3.3 Order calibration

A first-order role Markov model is compared with an IID role model on held-out manuscripts. The same model is scored after within-recipe role shuffling.

## 4. Frozen gates

### Lexical role gate

Pass only when:

- eligible-role macro-F1 is at least 0.60; and
- at least three eligible roles, or every eligible role when fewer than three exist, individually attain F1 at least 0.40.

### Identity-neutral role gate

Pass only when:

- structural-HMM eligible-role macro-F1 is at least 0.35; and
- its margin over the held-out majority baseline is at least 0.10.

### Role-order gate

Pass only when:

- mean real-order advantage over within-recipe shuffling is at least 0.05 bits/token; and
- the advantage is positive in every fold.

### Downstream admissibility

The sealed Voynich transfer stage is admissible only if all three gates pass. Thresholds are not revised after output inspection.

## 5. Secondary task

Recipe-level type recovery is reported for the frozen classes `RECIPE`, `MEDICINAL`, `TECHNICAL`, and `TIP`. A class is evaluated only with at least 20 units from at least three manuscripts. This task is diagnostic and does not control downstream admissibility.

## 6. Interpretation

A pass would establish that a real medieval procedural corpus contains recoverable operational roles under manuscript holdout, thereby providing a valid calibration target for Voynich. It would not imply that Voynich is a recipe collection.

A failure means that this representation/model cannot authorise semantic-role transfer. The correct result remains calibration failure, not a forced nearest-role account.
