# Amadi Babuini v1 — Programme Closeout

Date: 2026-08-11
Branch: `experiment/amadi-babuini-v1-20260811`
Status: **STOPPED AT PREREGISTERED BAB_H1 DECISION POINT**

## Executive result

The programme found **no admissible Voynich candidate** for the deterministic core Amadi Babuini mechanism.

The exact-source investigation materially corrected the initial model before target scoring. The 61 signs described in the solved example are the 61 distinct signs *used in that ciphertext* (14 ordinary letters + 47 babuini), not the complete codebook. The core source architecture instead assigns individual signs to CV syllables and separately supplies ordinary letter signs. Under the inherited 19-letter normalization this yields a finite 89-unit core model.

A literal attempt to reconstruct those signs from adjacent RF characters failed synthetic qualification and was abandoned without target scoring. Moving to the existing Zandbergen full-STA / connected-aaa representation solved the representation problem: top-89 vocabularies selected on FIT-A retain >99.5% whole-word character coverage.

The direct 89-sign decipherment instrument then qualified extremely strongly. On BAB_H1, however, both representation fits failed the preregistered optimizer-convergence gate and both held-out scores were far below the frozen natural-message floor. Therefore the result is **UNRESOLVED_SEARCH / NO CANDIDATE**, not a formal negative proof.

BAB_C1 remains sealed.

## 1. Source correction

The supplied Amadi/Scheers analysis supports:

- core Babuini: consonant+vowel syllable -> unique cipher sign;
- separate ordinary-letter transposition signs;
- q-vowel exceptions such as `qua`, `que` as three-letter babuini;
- a solved example using 61 distinct signs, which must not be mistaken for total key-sheet capacity;
- a separate, later 1,365-combination expanded syllabary covering richer syllable forms.

The v1 61-unit approximation was therefore superseded before any Voynich holdout score.

## 2. Representation ladder

### RF adjacency reconstruction — calibration blocked

The source-corrected core contains 89 normalized plaintext units. An engineering attempt represented these as 19 singleton RF signs plus 70 learned adjacent-pair compounds. On fresh synthetic Babuini controls this recovered only about 5% of hidden units and failed convergence.

Verdict: **RF PAIR UNITISER CALIBRATION BLOCKED**.

No target score was generated under this representation.

### Full STA / connected aaa — structurally viable

A FIT-A-only census using the frozen RF1b STA source and official `bitrans` conversion found:

- full STA: 158 observed FIT-A types; top 89 retain 0.9993815 occurrence coverage and 0.9976558 whole-word character coverage;
- connected aaa: 129 observed FIT-A types; top 89 retain 0.9997247 occurrence coverage and 0.9988711 whole-word character coverage.

This cleared the registered >=0.995 gate for both representations without inventing glyph grouping.

## 3. Qualification

Formal job: `6a7b63c5f6d0f3ee953a9f18` — completed.

Fresh namespace: `AMADIBABUINISTAV1Q1`.

Twelve independent 18,000-unit fit + 18,000-unit held-out controls were encrypted under independent hidden 89x89 keys.

Results:

- 12/12 converged;
- every control converged at the minimum six restarts per ensemble;
- median held-out unit recovery: **0.99997223**;
- minimum held-out unit recovery: **0.99966674**;
- minimum A/B map agreement: **1.000000**;
- structured-negative false positives: **0/60**;
- frozen absolute positive-control floor: **-3.1594953706 nats/Babuini-unit**.

The deterministic core Babuini instrument is therefore strongly qualified.

## 4. BAB_H1 target

Target job: `6a7b649b27caad61c6eac23e` — completed.

BAB_H1 contains 11 folios prospectively split from the previously untouched Amadi C2. BAB_C1 contains the other 12 and was not read by the release target runner.

### Full STA

- FIT-A whole-word character coverage: **0.99765581**;
- BAB_H1 coverage: **0.99834254**;
- fit score: **-4.30162445**;
- BAB_H1 score: **-4.27958842**;
- frozen floor: **-3.15949537**;
- deficit to floor: **-1.12009305 nats/unit**;
- A/B map agreement: **0.64379473**;
- convergence: **FAIL**.

Verdict: **UNRESOLVED_SEARCH**.

### Connected aaa

- FIT-A whole-word character coverage: **0.99887114**;
- BAB_H1 coverage: **0.99867330**;
- fit score: **-4.35224342**;
- BAB_H1 score: **-4.37858131**;
- frozen floor: **-3.15949537**;
- deficit to floor: **-1.21908594 nats/unit**;
- A/B map agreement: **0.48377494**;
- convergence: **FAIL**.

Verdict: **UNRESOLVED_SEARCH**.

## 5. Scientific interpretation

The two held-out scores are not marginal misses. They are more than one nat/Babuini-unit below the weakest qualified natural-message control, independently in full STA and connected aaa. Descriptively, that is strongly adverse to the deterministic core-Babuini hypothesis.

But the target map ensembles did not converge. The preregistration explicitly forbids turning a nonconverged optimization result into negative evidence. Increasing restarts or changing the optimizer now would be post-H1 rescue and is not permitted under v1.

Accordingly the binding result is:

**AMADI CORE BABUINI v1: NO POSITIVE VOYNICH CIPHER EVIDENCE; NO H1 CANDIDATE; FORMAL FAMILY REJECTION WITHHELD BECAUSE TARGET OPTIMIZATION DID NOT CONVERGE.**

No representation reached the candidate gate, so BAB_C1 remains sealed.

## 6. Expanded 1,365-combination Babuini

The later Amadi sections 386–389 establish a much larger 1,365-combination syllable inventory. That is not equivalent to the deterministic CV-core model tested here.

Under the supplied source extraction, the exact target-executable rule needed for a prospective test is not fixed: in particular, the source evidence available here does not determine one unique segmentation/use schedule between ordinary letter signs and the many overlapping longer syllable signs. Inventing such a schedule after seeing BAB_H1 would make the family non-identifiable and post-hoc.

Status: **SOURCE_UNDERDETERMINED / NOT TARGET-SCORED**.

It can be reopened only if the historical source or an independent reconstruction fixes a deterministic or tightly bounded syllabification/use rule before any further Voynich holdout is touched.

## 7. Relation to earlier Tranchedino syllabary work

Tranchedino v2.4 B1-O1 failed its registered optional-use family because low syllabary-use strata were not identifiable. At high use (0.75–1.00), recovery was near exact. The Amadi core programme therefore did not simply repeat that failure: it prospectively tested the narrower deterministic CV-use mechanism and successfully qualified it before target scoring.

## 8. Reopening conditions

Core Babuini may be revisited only with a materially new inference algorithm frozen and qualified on fresh controls before any reuse of BAB_H1/BAB_C1, or with independent manuscript evidence fixing a different surface representation or key structure.

Expanded Babuini may be tested only after an exact historical segmentation/use schedule is independently fixed.

Merely increasing the v1 annealing budget, selecting another STA vocabulary after H1, introducing RF bigram compounds, or inspecting the sealed BAB_C1 is not a valid continuation.

## 9. Compute closeout

- v1 61-sign smoke: `6a7b5ffdf6d0f3ee953a9ed4` — completed;
- core RF-pair smoke: `6a7b613bf6d0f3ee953a9ee4` — completed;
- STA/aaa representation census: `6a7b628ff6d0f3ee953a9ef7` — completed;
- direct-sign development: `6a7b6303f6d0f3ee953a9f15` — completed;
- manifest freeze: `6a7b634627caad61c6eac21a` — completed;
- formal qualification: `6a7b63c5f6d0f3ee953a9f18` — completed;
- BAB_H1 target: `6a7b649b27caad61c6eac23e` — completed.

Final Hugging Face process check: no running jobs.

No paid job was left orphaned.
