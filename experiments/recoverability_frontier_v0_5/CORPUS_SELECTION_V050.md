# Recoverability frontier v0.5.0 — corpus selection

Date: 2026-07-15

Status: **stage-0 acquisition and licence audit**

No Voynich text is used.

## Purpose

The pilot requires corpora that are reproducible, sufficiently large, script-diverse, typologically non-identical, and already separated into train, development and test partitions. Universal Dependencies is used as the scalable base because its repositories provide standardised CoNLL-U sentence text, explicit licences, fixed splits and broad language coverage.

## Frozen pilot languages

| Language | Treebank | Typological purpose | Pinned commit |
|---|---|---|---|
| English | UD English EWT | mostly analytic fusional baseline | `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b` |
| German | UD German GSD | fusional morphology and compounding | `ce54dbe9c6a5640c93e9952f069f582f6cd1f9fc` |
| Finnish | UD Finnish TDT | agglutinative morphology and rich case | `bfaae13719f249573d940edda6a0d7aa8eec620f` |
| Turkish | UD Turkish IMST | agglutination and vowel harmony | `0c939115d8277ecfb39e1bbc3f066b1852ab5ddc` |
| Hebrew | UD Hebrew HTB | Semitic root-pattern morphology and abjad script | `dd6d2133e6b9373e7e4888a1b33724df38e2e549` |
| Arabic | UD Arabic PADT | Semitic root-pattern morphology and abjad script | `dfb6b4c547f1fe10f1857b39e44de3f86c47a2fe` |

The selection is intentionally not optimised for Voynich historical plausibility. Its purpose is to falsify whether a decoder generalises across materially different linguistic systems. Historical Greek, Latin and medieval vernacular corpora enter only after the general recovery architecture passes this pilot.

## Leakage boundary

- Source language models and decoder training may use only each treebank's `train` partition.
- Hyperparameter and abstention calibration may use only `dev`.
- Scientific outcomes are reported once on `test`.
- Sentences are grouped into deterministic non-overlapping chunks; no sentence may occur in more than one trial.
- Corpus files are downloaded by immutable commit and verified against the stage-0 SHA-256 manifest before use.

## Licence boundary

Stage 0 records the exact licence file and hash for every repository. Any corpus without an explicit licence compatible with this research use is excluded before the cipher programme is frozen.

## Scale path

The loader is repository-driven rather than language-specific. After the six-language gate, the same acquisition layer can expand to dozens or hundreds of UD languages by adding pinned repository records without changing the scientific scoring code.
