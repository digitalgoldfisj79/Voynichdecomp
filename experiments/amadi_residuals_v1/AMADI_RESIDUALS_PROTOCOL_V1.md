# Amadi Residual Cipher Exploration Programme v1

Date drafted: 2026-08-11
Branch: `experiment/amadi-residuals-v1-20260811`
Parent: Cipher Coverage Closeout v1 @ `418da5635ffa2b1e86053dfd49fc1022ba15c297`
Source ledger: `SOURCE_LEDGER_V1.md`
Status: **PROGRAMME DEFINED; NO AMADI-RESIDUAL VOYNICH TARGET SCORE MAY BE COMPUTED UNTIL THE PRE-TARGET FREEZE IS COMPLETE**

## 1. Scientific question

The previous closeout reached a legitimate stopping point for the finite, historically justified circa-1400–1450 mechanisms then admitted. The complete Amadi catalogue exposes later-Renaissance finite mechanisms that are materially distinct from some of those tested families.

This programme asks two separate questions:

1. **Historical question:** can any Amadi residual mechanism be securely traced to an operational witness at or before 1450?
2. **Computational question:** irrespective of chronology, can a bounded Amadi-derived mechanism be prospectively recovered on fresh controls and is the frozen Voynich RF text compatible with it on held-out data?

The answers must never be conflated. A computational fit to a mechanism securely attested only in the late sixteenth century does not make the mechanism historically plausible for Voynich. Conversely, a historically plausible mechanism is not evidence unless the inference instrument qualifies and the held-out target passes absolute and mechanism-specific gates.

## 2. Relationship to the prior closeout

This programme does **not** nullify the previous result.

The previous statement is restated more precisely as:

> Broad identifiable cipher mechanisms with a defensible circa-1400–1450 anchor, under the families and representations admitted by Cipher Coverage Closeout v1, are closed as an active programme unless new external evidence or a newly validated inference method reopens them.

Amadi residuals begin as a **later-Renaissance stress-test**. Only an independent historical grade `H0/H1` can reopen the earlier chronological coverage claim.

No M0/TQ/NQ target result may be retuned or rerun as part of this programme.

## 3. Hard anti-fishing rules

1. Mechanisms are selected from the Amadi source ledger, not from Voynich score behaviour.
2. The primary target representation remains Zandbergen Reference Transliteration reduced EVA `RF1b-er.txt`, with the same source SHA and core-19 parsing policy used by the prior closeout.
3. No STA/full-STA/aaa/alternative transliteration may rescue a failed primary result. Alternative representation is replication only after a fully admissible candidate.
4. No favourable folio, section, Currier, hand or illustration subset may rescue a failed whole-corpus family.
5. No per-section, per-folio, per-line or free per-word keys.
6. No free token segmentation, arbitrary glyph-class collapse, arbitrary codebook, or target-derived alphabet reduction.
7. No plaintext inspection from a target fit until all positive-candidate gates pass.
8. No family deletion after target scoring because a difficult rule hurts family qualification.
9. No family addition after target scoring. A genuinely new source mechanism requires a new protocol version.
10. A solver that fails positive controls blocks inference; it does not count against the cipher family.
11. Beating shuffled data alone is insufficient. Absolute positive-control compatibility and mechanism-specific advantage are required.
12. Historical chronology is frozen before target scoring and cannot be upgraded because a target result is attractive.

## 4. Programme architecture

The programme has five stages and two optional conditional arms.

### Stage H — historical provenance audit

For every mechanism in `SOURCE_LEDGER_V1.md`, establish:

- exact Amadi section/folio;
- operational description;
- whether a worked example survives;
- earliest secure operational attestation found outside Voynich literature;
- earliest claimed antecedent;
- primary/critical-edition/manuscript evidence;
- date and place;
- historical grade `H0/H1/H2/H3/HX`.

Search priorities:

1. primary manuscripts/facsimiles and critical editions;
2. library/manuscript catalogues;
3. specialist history-of-cryptography scholarship;
4. general secondary synthesis only for discovery.

Voynich forums, blogs and Voynich-specific papers may not establish chronology or mechanism admission.

The audit must explicitly investigate Arabic cryptological traditions, Italian diplomatic cryptography, Alberti/Bellaso antecedents, Trithemius, Lullian combinatorial traditions, and other earlier witnesses suggested by Amadi himself, but must not infer transmission merely from conceptual similarity.

Deliverable: `HISTORICAL_PROVENANCE_LEDGER_V1.md`.

### Stage S — source reconstruction and structural feasibility

Before building a solver:

- reproduce the exact source algorithm from the Scheers text and illustrations where possible;
- reproduce at least one worked example exactly for an `ATTESTED_EXACT` family;
- determine output alphabet cardinality, expansion/compression ratio, word-boundary behaviour, state/key requirements and invertibility;
- compare only these coarse structural invariants with the frozen RF census.

This stage may examine target alphabet size, word-length support and boundaries but may not compute any language/cipher likelihood.

Direct mechanisms that cannot attain >=0.995 retained-character coverage without an unlicensed latent representation are labelled `SURFACE_INCOMPATIBLE_EXACT` and stop.

Expected examples:

- literal NTR/NTRC/DBAC and literal modulo-105 are likely to face this gate because their exact output alphabets are tiny;
- a post-hoc collapse of nineteen RF symbols into 3–7 latent classes is not permitted as an exact-mechanism rescue.

Deliverables: `SOURCE_RECONSTRUCTION_V1.md`, `STRUCTURAL_GATE_V1.json`.

### Stage Q — inference qualification on fresh controls

No Voynich cipher score is permitted until a family passes the relevant qualification gates.

### Stage V — sealed Voynich compatibility test

Only qualified families are opened on the target. Negative and positive decision rules are frozen in advance.

### Stage C — exact-source narrowing / confirmation

A broad superfamily that survives Stage V must be narrowed to source-exact schedules before any historical interpretation. Final confirmation uses a target subset that remains unscored by this programme until all prior gates pass.

## 5. Primary admitted families

### F1 — `R12H`: reduced alphabet + bounded homophonic surface

Source anchor: sections 024–026; related reductions 390–392 and 454–458.

Purpose: test the information-losing reduction step that M0 did not cover.

Primary language: Italian only, because the exact reduction rules in the present source are Italian. Other languages may be admitted before target scoring only when an independent historical source supplies an explicit reduction rule.

Model:

1. apply one exact Amadi reduction rule to plaintext;
2. the reduced plaintext alphabet contains exactly the source-defined reduced values;
3. map the 19 RF core symbols globally onto the reduced values by a surjective many-to-one mapping;
4. every RF symbol has one fixed latent value document-wide;
5. every latent value has at least one RF symbol;
6. no section- or position-dependent remapping.

Why the homophonic surface is admitted: section 026 explicitly combines reduced-alphabet thinking with multiple cipher signs for some reduced values. The 19-to-12 stress-test is broader than any one worked Amadi table and is therefore a conservative compatibility superfamily, not an exact historical reconstruction.

Lossy-recovery metric: accuracy is measured against the **forward-reduced plaintext surface**, not the unrecoverable original distinctions.

Exact subrules to freeze at source reconstruction: Vol-1/024, Vol-3/390, Vol-4/454 as and only as supported by the transcription.

### F2 — `VC_END`: vowel/consonant-dependent word transform

Source anchor: section 013.

Forward transformation per word:

`plaintext -> consonants in original relative order || vowels in original relative order`

followed by one global monoalphabetic relabelling.

Because the original vowel positions are destroyed, inference trains and scores a language model on the same **forward-transformed corpus**. No fictitious unique inverse is used.

Vowel classes are fixed from the normalized language corpus before controls. For the inherited Latin-script normalization the base class is `{a,e,i,o,u}`; language-specific departures require source/corpus rules fixed before qualification.

No 438–440 variant is admitted until the exact source operation is reconstructed.

### F3 — `PWA_K`: word-reset positional multi-alphabet superfamily

Source anchors: sections 445 and 461; supporting principle in 459, 463, 478.

Purpose: test position-dependent substitution tables without allowing arbitrary per-word keys.

Frozen broad family:

- `K ∈ {2,3,4,5}`;
- each position class has one independent document-global bijective substitution map;
- word position class is `(character_index_from_1 - 1) mod K`;
- the phase always resets to class 0 at every certain word boundary;
- maps are global across all folios, sections and hands;
- no optional phase shifts.

This is deliberately broader than the exact Amadi schedules. If the best qualified member is incompatible, narrower source schedules are conservatively disfavoured. If it survives, Stage C must test the exact 445/461 schedule before any Amadi claim.

### F4 — `GHOUSE5`: predeclared visible-selector house hypothesis

Source anchor for houses: sections 486, 489–490.

Status: target architecture hypothesis, **not an Amadi-attested gallows function**.

Frozen target selector values: `{k,t,p,f,NONE}` using the RF/EVA gallows glyphs only.

Before protocol freeze, implementation must define a deterministic selector extractor that:

- assigns at most one selector class per word;
- never uses semantic or section information;
- handles multiple gallows deterministically;
- reports selector coverage and ambiguity;
- treats the selector as control metadata rather than silently scoring it as ordinary payload.

Each selector class owns one document-global bijective substitution map. No map varies below the document level.

Additional positive gate on target: the real selector assignment must beat a pre-frozen deterministic panel of 256 within-folio selector-label permutations preserving class counts. Passing an ordinary language floor without selector specificity yields `COMPATIBLE_NONSPECIFIC`, never a house candidate.

## 6. Conditional families

These are source-research tasks, not automatically admitted target arms.

### C1 — exact Amadi plaintext autokey

Section 490.

Admit only if Stage S can reconstruct one unique state transition from the source and reproduce the worked operation. A generic modern autokey is forbidden as a substitute.

If admitted, the state machine, initialization, reset policy and table generation must be frozen before qualification.

### C2 — exact walking/two-stream cipher

Sections 369 and 373–376.

Admit only if the exact output can be represented as a unique linear cipher stream under a fixed key and a historical worked example is reproduced. If the method fundamentally requires a cover text, spatial carrier or missing physical cue, direct RF testing stops.

## 7. Prospectively admitted composition

Only one composite beyond `R12H` is admitted in v1:

### `R12_PWA`

Source motivation: sections 456–458 explicitly combine twelve-letter reduction with polyalphabetic tables/wheels.

It may enter target testing only if:

- `R12H` qualifies independently;
- `PWA_K` qualifies independently;
- an exact source reconstruction shows the composition order and schedule;
- the composite passes its own fresh positive controls.

No other combination of failed or weak mechanisms is permitted.

## 8. Target representation and holdout preservation

Primary source remains exactly:

`https://www.voynich.nu/data/RF1b-er.txt`

Expected SHA-256 inherited from the prior closeout:

`eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2`

Core-19 alphabet and uncertainty/word-boundary rules are inherited unchanged from `experiments/cipher_coverage_closeout_v1/TERMINAL_CIPHER_PROTOCOL_V1.md`.

### Holdout policy

The earlier C1 subset was never scored in Cipher Coverage Closeout v1. To preserve as much new target information as possible:

- `FIT-A`: prior T1 + prior H1 folios; model/key fitting only;
- reconstruct prior C1 exactly using the frozen old split namespace;
- divide prior C1 into `H2` and `C2` by SHA-256 of `AMADIRESIDUALV1::<folio>`, first half H2 and second half C2 after hash sorting;
- `H2`: first Amadi-residual compatibility decision;
- `C2`: remains unscored unless an H2 candidate passes all specificity gates.

Before any H2 cipher score, write `TARGET_SPLIT_MANIFEST_V1.json` containing every folio and source hash.

This is a programme-relative holdout, not a claim that those folios have never been studied elsewhere in the broader Voynich project.

## 9. Plaintext corpora

Default general-language panel is inherited from the qualified terminal programme:

1. Latin — UD Latin ITTB;
2. Italian — UD Italian ISDT;
3. German — UD German GSD;
4. French — UD French GSD;
5. Ancient Greek — UD Ancient Greek Perseus;
6. Hebrew — UD Hebrew HTB;
7. Arabic — UD Arabic PADT;
8. Spanish/Castilian — UD Spanish AnCora.

The existing training/control sentence-residue split remains binding unless a pre-target source audit demonstrates a defect.

`R12H` begins Italian-only. `VC_END`, `PWA_K` and any exact invertible schedule use the eight-language panel after the same historical normalization as the terminal programme.

No web text, LLM-generated text, Voynich-adjacent reconstruction or target-derived vocabulary enters the language models.

## 10. Primary scorer

Use a fixed word-sensitive character language model with explicit word-boundary markers and order-3 character probabilities with add-0.25 smoothing/backoff to unigram support.

Scores are total held-out log likelihood divided by retained transformed plaintext units.

The scorer is frozen before Q1. Alternative n-gram orders or neural language models may not be introduced after seeing qualification or target outcomes. If the order-3 scorer fails controls, the family is calibration-blocked under v1; changing the scorer requires a new protocol version and fresh controls.

For lossy families, the model is trained on the **forward-transformed** corpus.

## 11. Solver classes

### Global bijection solver

Reuse/adapt the qualified dual-ensemble simulated-annealing machinery from `terminal_cipher_v1.py` for `VC_END` and exact source schedules reducible to a global bijection.

### Multi-map solver

For `PWA_K` and `GHOUSE5`:

- maintain one bijective cipher->plaintext map per fixed state;
- initialize each map independently by within-state frequency rank;
- optimize pair swaps within one state at a time against the common held-out language objective;
- use two independent A/B ensembles;
- adaptive restart batches stop only when objective and occurrence-weighted map agreement converge;
- maximum restart budget is frozen before Q1 and may not be enlarged after target scoring.

### Reduced-homophone solver

For `R12H`:

- each of 19 observed symbols owns exactly one latent reduced value;
- assignments are globally fixed;
- optimization moves one observed symbol between latent values while preserving surjectivity;
- initialization and proposals are deterministic under recorded seeds;
- two independent ensembles must converge on transformed plaintext score and occurrence-weighted latent assignment.

No solver may inspect plaintext from target output.

## 12. Qualification gates

### Q0 — source fidelity

For each exact family/rule with a surviving worked example:

- reproduce forward encryption/output exactly after documented editorial normalization;
- reproduce inverse where the source mechanism is invertible;
- record any discrepancy rather than silently correcting Amadi/Scheers.

Gate: exact reproduction or an explicit editorial-error reconciliation supported by the edition. Otherwise `SOURCE_UNDERDETERMINED`.

Broad stress-test superfamilies (`PWA_K`, `R12H` 19-to-12 surface) must instead reproduce their own generated controls exactly and remain explicitly labelled broader than source-exact.

### Q1 — oracle-rule recovery

Fresh positive controls are disjoint from scorer training and from all previous cipher programmes.

Per exact rule/state count:

- minimum 3 independent controls;
- minimum 1,200 transformed plaintext units in fit and 1,200 in holdout per control;
- fresh keys/maps per control;
- two independent optimizer ensembles.

Binding gates:

- all controls converge;
- median transformed-plaintext recovery >=0.95;
- minimum recovery >=0.85;
- median occurrence-weighted A/B mapping agreement >=0.95;
- minimum agreement >=0.90.

For a lossy transform, recovery is scored against the known forward-transformed string.

A family with any frozen rule failing Q1 is `CALIBRATION_BLOCKED_Q1` and does not reach target scoring.

### Q2 — blind rule/family/language recognition

Construct a fresh balanced set not used in Q1. The solver does not receive the true family member, rule or language.

Binding gates across the admitted family universe:

- family accuracy >=0.90;
- exact-rule accuracy >=0.85 where rule identity is meaningful;
- language accuracy >=0.90;
- median transformed-plaintext recovery >=0.90;
- no language with >=4 controls may have accuracy below 0.75.

`R12H` Italian-only is judged on exact reduction-rule recognition and recovery rather than multilingual ranking.

### Q3 — absolute positive calibration

For every qualified `family × language` cell, generate 8 fresh positive controls under deterministically balanced rules/keys.

For each control:

- fit only on its fit partition;
- freeze maps/state parameters;
- score held-out positive data;
- also fit/score the appropriate simpler baseline, normally M0.

Freeze two floors per cell:

1. `ABS_FLOOR`: linear 5th percentile of the 8 true-family held-out scores;
2. `DELTA_FLOOR`: linear 5th percentile of `(true-family score - simpler-baseline score)`.

A target result can be mechanism-specific only if it reaches both floors.

### Q4 — structured-negative specificity

Before target scoring, test at least 80 structured negatives, balanced across:

1. iid unigram matched to symbol frequencies;
2. order-2 Markov;
3. motif-repeat/mutate;
4. copy-mutate local process;
5. slot-grammar generator matched to word-length and positional support.

Negatives receive the same family search and held-out decision procedure as positives.

Binding gates:

- false-positive rate against `ABS_FLOOR` <=2/80;
- no generator class >1 false positive;
- no family may disproportionately absorb structured negatives without being separately marked `NON_SPECIFIC`.

Q4 is mandatory before a positive Voynich interpretation. Q1–Q3 are sufficient to support a held-out **negative incompatibility** if a converged target fit falls below its absolute floor.

## 13. H2 Voynich decision

For each qualified family:

1. fit all frozen family members/languages on `FIT-A` only;
2. freeze the best converged fit by fit objective under the preregistered selection rule;
3. score that frozen fit on H2;
4. compare to `ABS_FLOOR` and `DELTA_FLOOR`;
5. emit no plaintext.

Outcomes:

### `CLOSED_NEGATIVE_INCOMPATIBLE_V1`

Allowed when:

- representation coverage >=0.995;
- winning fit converged under A/B rules;
- H2 score < corresponding `ABS_FLOOR`.

### `COMPATIBLE_NONSPECIFIC`

Use when H2 reaches `ABS_FLOOR` but fails one or more of:

- `DELTA_FLOOR`;
- Q4 specificity;
- exact-source narrowing;
- family-specific gate.

No plaintext inspection and no C2 opening.

### `H2_CANDIDATE`

Requires all of:

- H2 >= `ABS_FLOOR`;
- mechanism advantage >= `DELTA_FLOOR`;
- Q4 passed;
- solver converged;
- source-exact or explicitly broad-superfamily status stated;
- all family-specific gates passed.

Only then may Stage C and C2 be opened.

## 14. Family-specific target gates

### `R12H`

A candidate must additionally show:

- stable 19->12 latent assignment across A/B ensembles;
- no latent class supported only by negligible rare symbols;
- exact Amadi reduction rule selected prospectively or by blind Q2, not chosen after target score.

### `VC_END`

A candidate must beat M0 by the frozen `DELTA_FLOOR`; merely producing a language-like transformed surface is insufficient.

### `PWA_K`

A candidate must:

- identify one K under blind controls reliably;
- beat a matched non-word-reset positional model or phase-shuffled control at the calibration-derived threshold;
- proceed to exact 445/461 narrowing before C2.

### `GHOUSE5`

A candidate must:

- pass Q1–Q4;
- beat the 99th percentile of 256 deterministic within-folio selector-label permutations on H2;
- maintain map stability within each selector class;
- be described only as support for the predeclared gallows-selector architecture, not as proof of semantics.

## 15. C2 confirmation

C2 remains unscored until one H2 candidate completes Stage C.

Before opening C2:

- write a candidate-specific confirmation protocol containing the exact family, language, rule, keys/maps-fitting procedure, thresholds and any exact-source narrowing;
- no parameter is learned on H2 beyond the predeclared model-selection step;
- refit on `FIT-A + H2` only if that refit rule was specified in the candidate confirmation protocol; otherwise keep the FIT-A map frozen.

A confirmed candidate must pass its C2 absolute and mechanism-specific floors without plaintext inspection.

Only after C2 confirmation may plaintext be emitted for human inspection, and any apparent reading is then a secondary validation problem rather than part of the selection criterion.

## 16. Historical interpretation after computational result

Report computational and historical status on separate axes.

Examples:

- `COMPUTATIONALLY_CLOSED / H3_LATE_ONLY`;
- `H2_CANDIDATE / H3_LATE_ONLY`;
- `C2_CONFIRMED / H1_PRE1450_ATTESTED`.

A later-only confirmed mechanism would be scientifically interesting but would create a chronology/transmission problem rather than solve it.

## 17. Reopening and stopping rules

Stop the programme when every ledger mechanism is assigned one of:

- `HISTORICALLY_SCREENED + COMPUTATIONALLY_CLOSED`;
- `SURFACE_INCOMPATIBLE_EXACT`;
- `CALIBRATION_BLOCKED`;
- `SOURCE_UNDERDETERMINED`;
- `NON_IDENTIFIABLE`;
- `H2_CANDIDATE`;
- `C2_CONFIRMED`.

Do not invent additional compositions to avoid a stopping point.

A new v1.x protocol is justified only by one of:

1. a newly verified historical source supplies an exact operation not in the ledger;
2. a conditional Amadi arm becomes uniquely reconstructable from source evidence;
3. a new inference method passes fresh controls without Voynich development;
4. independent manuscript evidence fixes a selector, key, route, segmentation or latent class before target inspection.

## 18. Compute discipline

Default to local/ordinary CPU qualification first. Paid external compute is used only where profiling shows it is useful.

Before every paid launch:

- inspect existing jobs;
- record job ID, hardware, commit SHA, mode and timeout;
- impose a finite timeout;
- do not launch duplicate speculative jobs.

After completion, failure or scientific stop:

- cancel any remaining job immediately;
- verify the provider job list is clean;
- archive the scientific result before the next target stage.

No long-running orphan jobs are acceptable.

## 19. Required repository outputs

Before first Voynich score:

- `SOURCE_LEDGER_V1.md` — present;
- `HISTORICAL_PROVENANCE_LEDGER_V1.md`;
- `SOURCE_RECONSTRUCTION_V1.md`;
- `STRUCTURAL_GATE_V1.json`;
- `TARGET_SPLIT_MANIFEST_V1.json`;
- `QUALIFICATION_RESULT_V1.json`;
- frozen executable source and SHA-256;
- `PRETARGET_FREEZE_V1.md` with exact family/rule inventory and thresholds.

After target:

- `H2_TARGET_RESULT_V1.json` + human-readable report;
- candidate-specific confirmation protocol if needed;
- `C2_TARGET_RESULT_V1.json` only if opened;
- `AMADI_RESIDUALS_CLOSEOUT_V1.md`.

## 20. Binding programme-level claim

This programme is designed to answer a narrower and more defensible question than 'is Voynich a cipher?':

> After explicitly adding finite later-Renaissance mechanisms exposed by the complete Amadi catalogue, and separately auditing whether any has a genuine <=1450 antecedent, does any prospectively recoverable mechanism remain absolutely compatible with held-out Voynich text?

A negative answer closes the Amadi blind spot without weakening the earlier methodology. A positive answer creates a precisely bounded candidate whose computational evidence and historical chronology can then be investigated independently.
