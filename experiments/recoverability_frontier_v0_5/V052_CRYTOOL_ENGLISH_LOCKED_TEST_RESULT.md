# Recoverability frontier v0.5.2 — CrypTool-style English locked test

Date: 2026-07-15

Verdict: **FAIL LOCKED MEAN-RECOVERY GATE; BASIN-HIT RELIABILITY IS THE BOTTLENECK**

No Voynich text was scored.

## Frozen test

- algorithm: strict fixed-inventory CrypTool-style exhaustive pair sweeps;
- schedule: 3,000,000 proposals × 12 independent restarts;
- target initial acceptance: 0.05;
- language: English;
- length: 384 normalized characters;
- untouched test replicate block: 96–115;
- trials: 20.

Job: `Digitalgoldfish79/6a58137885d9643ce16d5997`

Scientific SHA-256: `d5411c02edd5d8118162be1513399d8b87d8a90aa6a20371a006f131655d5191`

Complete row-level JSON is preserved as gzip/base64 in the immutable job log.

## Result

- mean recovery: **29.3620%**;
- median recovery: **11.0677%**;
- frequency baseline: **27.6172%**;
- exact recovery: 0%;
- inferred inventory overlap: **93.7833%**, preserved exactly.

The result is strongly bimodal:

- 4/20 trials recovered at least 90%;
- those four recovered 95.31%, 98.44%, 99.48% and 98.70%;
- the remaining 16 trials remained in incorrect basins, mostly near 8–18%.

## Interpretation

When the search reaches the correct basin, plaintext recovery is essentially complete. The failure is the probability that 12 independent trajectories discover that basin.

If each restart has approximately independent success probability `p`, an observed 20% per-trial success rate after 12 restarts corresponds to `p` near 1.9% per restart. Under that rough model:

- 48 restarts would yield about 60% basin-hit probability;
- 96 restarts about 83%;
- 192 restarts about 97%.

This estimate is descriptive, not a replacement for empirical development selection.

## Required next step

Do not alter the objective, inventory, temperature or proposal mechanism. Increase only the number of independent strict fixed-inventory restarts on development data. A future validation must use a new untouched test block.
