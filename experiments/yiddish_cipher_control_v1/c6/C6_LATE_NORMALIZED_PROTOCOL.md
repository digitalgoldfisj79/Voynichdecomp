# C6 late normalized transfer protocol

Status: **FROZEN BEFORE SOURCE OUTCOME**  
Date: 2026-09-12

Purpose: test source-family transfer of the already-qualified v0.2 M0 mechanism instrument **without changing its representation, training pool, objective, search budget, thresholds, or solver code**.

This is deliberately a limited certificate. It does **not** qualify 15th/16th-century transfer and does **not** qualify primary Hebrew-script/diplomatic representation.

## Frozen fresh source

Penn Historical Yiddish work: `1783e-ukraine-1.psd`.

Freshness checks completed before outcome exposure:
- no occurrence in the canonical Supabase Yiddish/cipher handoff state;
- no occurrence in `digitalgoldfisj79/Voynichdecomp` code search;
- not used in the prior R/L/full-control source lists recorded in `CONTROL_STATE.md`.

If an exact 8-word overlap exists between BUILD and this source, C6 fails contamination preflight before solver scoring. Exact 5-word overlaps are diagnostic.

## Representation and solver

Exactly the C0-C5 qualified secondary representation:
- Penn historical-Yiddish Romanisation;
- literal a-z atoms only under the existing extractor;
- preserved word order/boundaries;
- alphabet size 26.

Exactly the qualified solver:
- first-occurrence ciphertext-symbol canonicalisation;
- S1E frozen configuration: 8 restarts, 5000 search steps, 60 greedy passes;
- existing Yiddish BUILD training pool from the full control;
- same bigram objective;
- no source-specific tuning.

## Blinding

Preparation writes separate public and private bundles.

Public bundle contains:
- training words;
- ciphertext cases;
- source ID/hash/quantity and pre-outcome manifest;
- plant-root commitment only.

Private bundle contains:
- plaintext fit/audit truth;
- planted inverse keys;
- plant-root reveal.

Solver receives public bundle only. Truth is joined only after mappings are committed.

## Cases

Registered primary C6 cells:
- fit length 512 words;
- 32-word buffer;
- audit length 512 words;
- erasure 0%; 32 independent planted keys;
- erasure 1%; 32 independent planted keys.

If the source is shorter than 1,056 extracted words, status is `C6_LATE_NORMALIZED_CORPUS_LIMITED` and no solver run is admissible.

Additional 3%/5% or longer windows are prohibited in this confirmation version; they were not needed for the prospectively qualified envelope.

## Success rule

Per trial:
- eligible atom recovery >= 0.90;
- eligible whole-word recovery >= 0.80.

Per cell:
- >=29/32 successful trials.

Both registered cells must pass.

## Permitted verdicts

- `C6_LATE_NORMALIZED_PASS`
- `C6_LATE_NORMALIZED_FAIL`
- `C6_LATE_NORMALIZED_CORPUS_LIMITED`
- `C6_LATE_NORMALIZED_CONTAMINATION_FAIL`
- `C6_LATE_NORMALIZED_EXECUTION_INVALID`

A PASS means only:

> The frozen M0 instrument transfers to one previously untouched 1783 Yiddish source in the same secondary normalized representation at 512 words and <=1% erasure.

It does not license Voynich target access. Early historical transfer, primary-script transfer, and Yiddish language certificate L remain separately unresolved.