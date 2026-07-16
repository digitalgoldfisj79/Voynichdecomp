# Recoverability frontier v0.5.4 — initial Stage A result

Date: 2026-07-16

Verdict: **FAIL INITIAL COMPONENT GATES; SEPARATE CODEBOOK IDENTIFIABILITY FROM RESIDUAL-KEY RELIABILITY**

No Voynich text was scored.

## Frozen condition

- English corpus `dev`, eight non-overlapping word-aligned chunks;
- approximately 384 normalized plaintext characters;
- fresh monoalphabetic key;
- fresh 24-word codebook sampled from the top 96 train words;
- character and code symbols jointly relabelled;
- first-occurrence canonical surface input.

Job: `Digitalgoldfish79/6a58639385d9643ce16d5d25`

Scientific SHA-256: `fe53211ea5bfdbf49815878b0bbab6ae23cadba5f2faf37323056918300bf1bf`

The complete row-level artifact is preserved in the immutable job log.

## A1 — true character key, unknown code words

- mean observed code symbols: **4.5**;
- mean code-symbol mapping accuracy: **29.8512%**;
- mean code-occurrence word accuracy: **44.3624%**;
- mean expanded character recovery: **97.1964%**;
- median expanded recovery: 97.5260%.

The high character recovery is not evidence of successful nomenclator decoding. Only a small fraction of the text was represented by observed code symbols, so wrong code words produced a small edit-distance penalty.

The primary failure is identifiability: a fresh 24-word codebook sampled from a 96-word pool yields only 2–7 observed code symbols in these 384-character chunks. A train-only word n-gram often prefers a plausible frequent word over the true code word.

**A1 gate: fail.**

## A2 — true code words, unknown residual character key

- mean baseline expanded recovery: 37.1452%;
- mean residual key accuracy: 61.9554%;
- mean expanded recovery: **65.9118%**;
- median expanded recovery: **99.4838%**.

Five of eight trials recovered 99.48–100%. Three remained near 7–11%. The result is sharply bimodal and therefore a search-basin reliability failure, not a generally weak objective.

**A2 gate: fail.**

## Required decomposition

### A1 identifiability frontier

Vary independently:

- plaintext length: 384, 768 and 1536 characters;
- candidate codebook pool: top 32, 64 and 96 train words;
- fresh codebook size: 16 for pool 32, and 24 for pools 64/96.

Do not condition codebook selection on the test plaintext. Report observed code-symbol count, occurrence count, mapping accuracy and expanded recovery for every cell.

This determines the minimum observation regime in which codeword identity is empirically recoverable.

### A2 search reliability

Keep the generator and oracle information unchanged. Increase only the residual-key search budget on development data:

- 700,000 iterations × 50 restarts;
- 1,200,000 iterations × 70 restarts.

No codeword or plaintext information beyond the frozen A2 oracle may be added.

## Boundary

Stage B joint recovery remains prohibited. The generator gate will not be relaxed merely because expanded character recovery is insensitive to a handful of wrong code words.
