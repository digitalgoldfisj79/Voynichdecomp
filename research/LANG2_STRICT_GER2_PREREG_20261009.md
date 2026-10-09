# LANG2 — Strict GER2 language-arm replication (2026-10-09)
Status: PREREGISTERED / execution gated by source audit and smoke.
Scope: This is an independent source-language branch. It does not change or supersede the active Voynich residual-recovery R4 controller.

## RETRACTED FINDINGS — permanently prominent
1. The reduced one-epoch C-LANG mini-NeuroCipher is NOT a qualified language discriminator: substituted-Latin positive control failed; all its Latin, Padua, Czech, Mandarin outputs are non-results.
2. German source-language identification was NOT achieved by GER2. BAV failed full replication; ALEM failed the frozen fold4 validation z>2 gate.
3. The old PGCS/S6 ranking (e.g. Circa instans over German) is representation-conditioned and is NOT qualified evidence for an upstream language. Never restore it as a source-language result.
4. No raw cross-language expected-edit / NLL score comparison is valid because candidate vocabularies, scripts and opportunity structures differ.
5. C generation's 66.96-SD mismatch is NOT evidence against German: C was Voynich-trained, not a German LM.

## Preregistered primary question
For the same Voynich exact-token forms, does *real* vocabulary from a historically plausible language beat its own uniquely anagrammed frequency-ranked vocabulary under the original full GER2 instrument, on held-out physical bifolia, with a replicated null-normalized effect?

This is a test of DIRECT MONOTONIC SURFACE WORD TRANSDUCTION only; passing is not decipherment or unique source identification. The multi-language common-normalisation hypothesis is a separate future arm, not a post-hoc reinterpretation of this arm.

## Exact frozen mechanism / sample / statistics
Source repository: digitalgoldfisj79/Voynichdecomp
- Solver: research/neurodecipher_acl2019_refactor.py @ e36031a9a6b8b67fcebb4d6f4af1c3753fad4287
- GER2 runtime: research/vms_neurocipher_ger2_cached_runtime_20261006.py @ 98b7324fc65aa9b14242920f6931a8d1256064df
- GER2 VMS corpus and physical folds cache @ a4a133f0635e0408266bc185a1697821ab79bd28
- ZLZI strict +P0; first two tokens of every physical line excluded.
- Physical bifolium assignment SHA256 e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888.
- Discovery folds 2/3; validation fold 4; sealed final folds 0/1. A type in both discovery and validation is excluded from strict-final set.
- Discovery VMS vocabulary 735 types; target training top 4,103 word forms; target final top 10,000 forms; MCF discovery demand 221. Validation 735 novel types; strict-final 735 never-exposed types.
- Solver EXACTLY 10 rounds x 150 epochs per M-step; 10-epoch checkpoint evaluations; 5 warmup steps; reg_hyper 0.5; eval_batch 48; optimizer and E/M steps unchanged.
- Four seed pairs: model 1234/scramble 9001, 2026/7001, 17/11003, 73/17011.
- Real and scramble each train FROM SCRATCH, identical solver settings. Scramble unique one-to-one anagram of the exact same ranked 10,000 surface forms, preserving each form's length and letter multiset. Record unchanged and collisions.
- Model selection uses ONLY fold4 validation's lowest expected-edit MCF cost.
- Primary sealed outcome: final_strict_novel.mean_edit_cost. Secondary: fold4 selected-checkpoint validation.mean_edit_cost. Lower costs are better.
- Compute for each candidate separately: delta_final = mean(NULL_final)-mean(REAL_final), sd_null_final=sample SD of four independent scramble runs, z_final=delta_final/sd_null_final. Analogously for validation.
- GER2 EXACT gate: REAL mean < NULL mean at both endpoints; z_final >2; z_validation >2; at least 3/4 seed-matched wins on each endpoint; direction repeats across physical selection/sealed folds. No exception or post-hoc threshold revision. If |z|<2, lead interpretation with “the metric does not resolve this.”
- Never rank languages by raw cross-language costs. A further tournament across within-language standardised effects REQUIRES a separately frozen, null-calibrated comparison and independently qualified positives; raw z ranks are descriptive, not source-language identification.

## Current source arms (only source corpus changes)
- LATIN_CI: 52,004 original tokens in CI pickle; exact source SHA256 377daa2a9c2403e6b8e10146a9c67f9e6b8144e959f232ec9096cdd0a4ae81f1. Sole Latin corpus, NOT independent Siena witness replication. Exact ASCII-alphabetic full GER2-sized top 10,000 vocab SHA256 e5b62f9d76bbbae21218b2d5c18249b0976c30aedaed42484e1d7e029490a364. CI source pin 67a73f80da2caefd6788de43c003701225833825.
- PADUA_MIXED: original Transkribus PAGE XML zip SHA256 63c6f70bc144994b1fe34e83cb11d51441e8c0502005d73154adbe9004fea0f2, 263 PAGE XMLs. Latin AND Venetian/Italian vernacular. It must NEVER be promoted as a clean Tuscan/Italian answer. This input is locally recoverable but remote corpus transport and exact tokenizer audit must be frozen BEFORE any inferential run.
- BAV / ALEM original GER2 existing data = reference controls, no changing their results. Do NOT spend to rerun them merely to complete a table.
- Clean independent 15th-c vernacular Tuscan, dated Latin medical witnesses, Old French/Franco-Italian and Czech/Slavonic require corpus admissibility review BEFORE adding any arm. Do not silently replace short witnesses with Classical, print-era, modern or mixed-register texts to fill fixed 10,000-type slots.

## Earlier actual adjudication, not to be superseded
BAV: final mean REAL 0.07190249, NULL 0.09130392; delta +0.01940142; NULL SD 0.01959026; z 0.9904. Validation delta +0.01912984; SD 0.02261723; z 0.8458. FAIL.
ALEM: final REAL 0.05902475, NULL 0.11297804; delta +0.05395330; NULL SD 0.02173829; z 2.4819. Validation REAL 0.05846413, NULL 0.11195154; delta +0.05348741; NULL SD 0.03387451; z 1.5790. FAIL frozen validation gate.
Full source: Supabase handoff key voynich_vms_neurocipher_ger2_phaseB_20261006.

## External validity firewall
GER2 calibrated historically against ReF15 German dialects, not language-blindly against every Latin/Italian source condition. GER2's German calibration does not establish equal *sensitivity* across languages. Before any linguistic interpretation, a separate cross-language known-positive audit is required with independent manuscript witnesses and matched held-out gold, OR a transparent status “unqualified for language comparison.” Absence of this audit cannot retroactively invalidate the narrow frozen GER2 real-vs-anagram outcomes; it limits their interpretation.

Corpus provenance, manuscript count, genre, era, normalization, hapax counts, language mixing, transcription noise and lexicon-size support are mandatory per arm. No target-vocabulary truncation below 10,000, no extra one-epoch shortcuts, no novel per-language capacity choices.

## LANG2-B multilingual normalisation: PARKED, explicitly distinct design
H0: one upstream language, variable genre/hand/section and common surface constraints.
H1: multiple upstream languages, ONE frozen renderer, with identifiable language-conditioned residuals beyond genre/hand/physical placement.
Qualification BEFORE Voynich: independent labelled multilingual texts, planted single-source and multi-source encodings, held-out sources/witnesses, matched wrong-language and unlabelled-section controls; blind null false positive and planted detection. Reject circularity, source-label leakage, hand/section confounding, arbitrary renderer capacity, representation dependence and decision-rule fragility. Require physical-bifolium holdout and ZLZI-vs-TTLI robustness before making a Voynich multilingual claim.
No tuning renderer or choosing mixture after seeing Voynich target outcome.

## Execution / recovery
Immutable full-run wrapper: research/lang2_ger2_identity_20261009.py @ commit 10e8d498c745f833223b58ac2fdbfb66c7602371.
Smoke job: 6ac92296fee2c9007017ac45.
One log terminal JSON per replica: VGER2_RESULT_JSON. Record exact HF job IDs, commits, corpus/vocab hashes and verdict in independent handoff, not the canonical residual R4 controller. Failures are engineering reruns ONLY when algorithm/data/seeds unchanged.
