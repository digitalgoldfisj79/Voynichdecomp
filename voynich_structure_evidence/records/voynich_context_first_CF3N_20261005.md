# Voynich context-first CF3N — bounded notation-strength real equivalence test — 2026-10-05

- key: `voynich_context_first_CF3N_20261005`
- status: completed
- created: 2026-10-05T20:39:35.391504+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# VOYNICH CONTEXT-FIRST CF3N — BOUNDED NOTATION-STRENGTH REAL TEST
Date: 2026-10-05
Status: COMPLETED / POSITIVE UNDER BOUNDED CALIBRATION
Programme: voynich_context_first_lexical_equivalence_preregister_20261005
Scientific HF job: 6ac40a08fbc85ba6823acabe
Code commit: 22743d57a18a90a65fec7cd18a309e442e3cf0ea
Script: research/context_first_CF3_notation_strength_20261005.py

## Scope / critical interpretation
This is NOT a general test that can reject ordinary-prose homophony.
CF0 established:
- ordinary Circa-Instans-strength equivalence recovery is not identifying enough: heldout precision 15/27=.5556;
- the same frozen rule has overwhelming power for strongly ordered notation-like source equivalence: 120/120 true pairs across eight ordered seeds; shuffled order 6/120=.05.

Therefore CF3N tests only whether real Voynich contains CONTEXTUAL EQUIVALENCE at least as clean/structured as the calibrated notation-strength regime.
A positive does NOT establish homophony, semantics, plaintext, notation, or Naibbe.

## Frozen selection rule
No current-token morphology, FORM, ED, DINO, Stolfi/Mauro, gallows or length used in pair selection.
Population strict +P0; physical line-start current tokens excluded.
Discovery physical folds 2/3.
Validation fold4.
Final untouched folds0/1.
Candidate exact token types satisfy discovery/validation/final occurrence thresholds from the preregistration.

For each eligible exact-token pair:
1. nuisance-residualized external-context fingerprints from neighbours at lags -2,-1,+1,+2;
2. discovery split-half cross-similarity folds2 vs3;
3. retain top 1% cross-similarity;
4. validation context-only A-vs-B distinguishability classifier, C=.3;
5. retain bottom 30% by classifier gain (most contextually indistinguishable);
6. greedily resolve overlapping pairs into disjoint pairs before final.
No parameter retuned on Voynich.

Final statistic:
pool the selected pair's external-context model vs keeping exact-token-specific context models.
Report heldout codelength gain, physical-bifolium SE, fold0/fold1 effects, matched random pair-partition null, exact-token positive control.

## ZLZI
Eligible types: 103.
Top-1% candidates: 53.
Raw selected: 16.
Disjoint frozen pairs: 10:
1 okal / otar
2 checkhy / chy
3 lkaiin / opchedy
4 otam / shy
5 aiiin / aiin
6 qokaiin / qokar
7 okeey / qokeey
8 cheey / kaiin
9 cheky / qotar
10 kar / lchedy

Pooled-vs-exact context gain:
+0.0102666607 bits/context-event.
Physical-bifolium SE = 0.0038874259.
Effect/SE = 2.64099 SD.
Fold0 +0.00738412.
Fold1 +0.01284739.
Matched-random partition null mean -0.00806351, SD .00702388; observed-vs-null = 2.60970 SD.
Exact-token positive control +0.03465345 bits/event / .00423368 = 8.18518 SD.
PRIMARY GATE PASS.

## ZLZB
Eligible 103; 53 candidates; 16 raw selected; 10 disjoint.
Independently selects the SAME TEN PAIRS as ZLZI.

Pooled-vs-exact +0.0103042954 bits/event.
Physical-bifolium SE .0038936123.
Effect/SE 2.64646 SD.
Fold0 +.00742596.
Fold1 +.01288127.
Matched-random advantage 2.66197 SD.
Exact-token control +.03455719/.00421936 = 8.19015 SD.
PRIMARY GATE PASS.

## TTLI
Eligible 95.
Candidates 45.
Raw selected 14.
Disjoint frozen pairs 11:
1 r / s
2 aiiin / aiin
3 cheedy / keedy
4 okal / otar
5 checkhy / cheody
6 al / y
7 dair / dor
8 oteey / qokeey
9 okey / otey
10 otal / qotar
11 otedy / oteedy

Pooled-vs-exact +0.0092459532 bits/event.
Physical-bifolium SE .0025898974.
Effect/SE 3.57001 SD.
Fold0 +.01125184.
Fold1 +.00714345.
Matched-random advantage 3.24865 SD.
Exact-token positive control +.03020018/.00427407 = 7.06590 SD.
PRIMARY GATE PASS.

## Replication / representation
ZLZI and ZLZB are essentially exact replications and independently select the same 10 pairs.
TTLI also passes strongly, shares aiiin/aiin and okal/otar exactly, and selects related but not identical pairs around qokeey/qotar and other forms.
This argues against the positive being a single-transcription artefact.

## Adjudication
SUPPORTED:
1. A context-first procedure, with current-token form completely hidden during selection, identifies small sets of exact token pairs whose context distributions pool better than separate exact-token estimates on untouched physical material.
2. The effect is >2 physical-block SD and positive on both final folds in ZLZI, ZLZB, TTLI.
3. The pooled advantage exceeds matched random pair partitions by >2 SD in all three.
4. Exact-token context positive controls remain strong (~7.1-8.2 SD).
5. ZLZI/ZLZB exact pair replication is unusually strong.

NOT ESTABLISHED:
- that selected pairs are homophones;
- that they encode one hidden source item;
- that Voynich is notation;
- that source identity causes the pair relation;
- that context classes predict the surface of unseen forms;
- any semantic equivalence.

Important surface observation after selection only:
Some pairs are visibly near-neighbours (aiiin/aiin; okeey/qokeey). Others are not obviously close (lkaiin/opchedy; otam/shy; kar/lchedy). This is descriptive only and was not used in selection.

## Programme consequence
CF4 is licensed.
CF4 must keep these frozen pair relations fixed.
Current final token form must be masked at class inference.
The decisive question is whether context-inferred class predicts the surface/whole-form structure of exact token types unseen in discovery/validation.
A CF4 failure would reinterpret CF3 as contextual/syntactic/topic equivalence rather than a source->surface bridge.
A CF4 pass would be the first direct evidence that an upstream context/source class predicts an unseen Voynich surface form without using current-token morphology.

