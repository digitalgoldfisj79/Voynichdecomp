# TScore documented-contract ingestion audit — v0.6

**Date:** 2026-07-25  
**Status:** unit-test objective completed with an explicit scope limitation.

## Source status

The published 2024 TScore paper describes a proof-of-concept representation for German lute tablature and an XML intermediate model carrying duration, string, fret, graphical-position, beam and prolongation information. The paper stated that open-source publication was in progress. No authoritative public repository or stable release of the official implementation was located during this closeout.

Accordingly, the accompanying test is **not** presented as validation of official TScore software.

Primary reference:

- Alexander König, Markus Lepper and Reinier de Valk, “Data Models of German Lute Tablature With TScore,” arXiv:2410.10259 (2024), https://arxiv.org/abs/2410.10259

## Implemented contract test

`test_tscore_documented_contract_v06.py` is an independently written, self-contained contract test derived from the paper’s documented data semantics. Its fixture is synthetic and does not reproduce a historical musical passage.

It tests:

1. documented duration and beam-marker semantics;
2. duration inheritance/carry behaviour;
3. symbol-to-string/fret coordinate mapping;
4. aligned duration and voice ingestion with exact onset calculation;
5. rejection of misaligned voice rows.

Execution:

```text
python test_tscore_documented_contract_v06.py
Ran 5 tests in 0.000s
OK
```

Script SHA-256:

`4221bef3f8245b260474f2ee3ca6acdf5a8601b16810f22eaea9a28237f03547`

Recorded output SHA-256:

`a28b5c7db629f9bf9a6088d3a08538b0c0b841bbb8d01aef4a671a3387184bb7`

## What is established

The planned recoverability infrastructure can ingest an aligned tablature/event contract with deterministic timing, string and fret semantics, and it rejects malformed alignment. This closes the small proof-of-concept/unit-test action in the v0.6 acquisition plan.

## What is not established

The test does not establish compatibility with an unreleased official parser, correctness on historical TScore files, or cross-manuscript recoverability. Those require either an official release or representative files supplied by the authors.
