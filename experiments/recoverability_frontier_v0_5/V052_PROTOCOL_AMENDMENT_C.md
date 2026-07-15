# Recoverability frontier v0.5.2 — protocol amendment C

Date: 2026-07-15

Status: fixed after the six-trial-per-language quadgram holdout, before confirmation execution.

## Observed holdout

The frozen quadgram solver selected `700,000 × 50` on development and achieved on untouched test replicates 20–25:

- overall mean recovery: 77.6042%;
- English: 97.5694%;
- Turkish: 57.6389%;
- exact recovery: 25.0%.

The overall and short-text gates passed, but Turkish missed the 60% per-language floor by 2.3611 points. With only six trials per language, the estimate is too coarse for a definitive family stop.

## Frozen confirmation

Run the identical solver, model, inventory search and `700,000 × 50` schedule on 20 new untouched test chunks per language beginning at replicate 32.

No development rerun, schedule change, objective change, alpha change or inventory adjustment is permitted.

Confirmation passes only if:

- combined English/Turkish mean recovery is at least 70%;
- each language reaches at least 60%;
- the result remains above the original fixed-inventory smoke.

Failure stops v0.5.2 before six-language expansion. Passing permits the originally planned six-language × three-length diagnostic using a separately untouched test block.
