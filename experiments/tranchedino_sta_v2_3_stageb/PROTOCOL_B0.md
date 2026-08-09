# Tranchedino × STA Historical Cipher Programme v2.3 — Stage B0 frozen protocol

Date: 2026-08-09
Namespace: `TRANCHSTA23B0`
Parent branch head at creation: `bc6b717fd0ce8d92f0004a2cea971fdccbb521ff`

## Scope

Stage B0 is a **pre-target source and representation gate**. It contains no Voynich language score, no target mapping fit and no decoded text. Its purpose is to establish whether the exact f.69v mixed inventory can be represented at useful occupancy in STA and whether the genuine historical Paduan word-level source used by the earlier payload programme can be recovered exactly enough to support a nomenclator calibration.

No later Stage B may run if B0 fails.

## Frozen historical model

Binding strict f.69v inventory: K=92 one-sign assignments:

- 36 alphabetic homophones;
- 11 geminate signs (`bb cc dd ff gg ll nn pp rr ss tt`);
- 7 null signs;
- 38 one-sign nomenclator entries.

The two uncertainty-qualified f.69v alphabet cells are excluded. No syllabary is admitted. No free state/key changes are admitted.

## B0.1 — RF full-STA representation census

Source: René Zandbergen RF1b STA1 IVTFF.

Expected RF1b SHA-256:
`81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`

Parse full STA members using the same character rule and hard-break conventions as Stage A. Sort members by whole-RF descending occurrence frequency with lexical tie-break. Freeze the top 92 as the primary Stage-B visible-sign vocabulary.

Required pre-target representation gate:

- RF source hash exact;
- parsed full-STA character total reproduces Stage-A census (157,254 under the binding parser);
- observed member types >=92;
- K92 occurrence coverage >=0.995.

If RF K92 coverage <0.995, verdict: `STAGE B REPRESENTATION GEOMETRY FAILED`; stop.

Independent transcription census (diagnostic robustness, no target score): run the same K92 frequency census separately on ZL3b and GC2a level-1 STA. Both must have >=0.995 K92 coverage. This is not a same-map replication test; it only establishes that a 92-member visible inventory is not peculiar to RF.

## B0.2 — genuine Paduan source recovery

The previous historical payload programme used a supplied fifteenth-century Paduan PAGE-XML transcription. Before any nomenclator or mixed-output controls are generated, recover the **word-level/line-level source**, not merely a letter n-gram model.

Required invariants from the archived prior programme:

- PAGE-XML files: 261;
- transcribed lines: 5,735;
- cipherable letters before the later 19-letter normaliser: 227,702;
- old LM/payload split reported 172,362 / 54,764 cipherable letters;
- the v2.0 19-letter reconstruction reported 4,119 LM lines / 172,347 retained characters and 1,423 held-out lines / 54,750 retained characters, chronological cut page 183.

The recovered source must preserve line boundaries and word boundaries and must reproduce the old chronological split. File/source hashes must be recorded before B1 design.

A checksum record without the underlying word-level source is insufficient.

If the exact source cannot be recovered, verdict:
`PADUAN WORD SOURCE NOT RECOVERED / STAGE B BLOCKED`.

No modern Italian, Latin, synthetic word list, language-model reconstruction or internet replacement may be substituted after this failure.

## B0.3 — source-only occurrence feasibility

Only after B0.2 passes, conduct a source-only census on the held-out Paduan partition. No Voynich data may be loaded.

Purpose: determine whether a 38-entry fresh nomenclator can be observed often enough at the intended control length to make codebook identity empirically recoverable.

Allowed outputs before B1 freeze:

- word-type and token-frequency distribution by train/held-out partition;
- expected distinct/total code occurrences for deterministic candidate-pool sizes and sample lengths;
- availability of 12 non-overlapping or deterministically sampled 12,000-letter controls;
- geminate occurrence counts for the 11 binding doubles.

This census may choose the **observation regime**, not solver thresholds or target parameters. B1 codebook-pool size and control length must be frozen from these source-only counts before any mixed-unit synthetic control outcome is inspected.

## Stopping rule

B0 contains no Voynich adjudication. Any representation/source failure closes v2.3 before solver construction. Passing B0 authorises a separate prospective B1 calibration protocol; it does not authorise target scoring.
