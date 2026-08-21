# SVT v0.2 — deterministic factorised key-search programme

Frozen: 2026-08-21, before v0.2 synthetic results.

## Why a new version

SVT v0.1/v0.1.1 is retained as a failed instrument-development lineage. Component diagnostics showed:

- exact truth state maps decode the synthetic plaintext at 100%;
- with truth maps and truth period, the language model selects periodic vs line-reset mode correctly in 4/4 diagnostic trials;
- supplying the true mode and true period but estimating the factorised key still recovered only 24.1% mean plaintext;
- therefore the immediate failure is the stochastic key optimiser, not ambiguity in the synthetic construction or mode definition.

The v0.1 stopping discipline is not relaxed. v0.2 is a separately versioned solver architecture for the same bounded FSVT family.

## v0.2 change

Replace stochastic state-local annealing with deterministic coordinate optimisation over the actual frozen factorisation:

1. estimate one shared monoalphabetic inverse key from the full head stream;
2. refine the shared key by exhaustive best-improving pair swaps applied to all states together;
3. for each state, exhaustively evaluate pair swaps of plaintext assignments and accept the best move only if its whole-sequence language-model improvement exceeds a fixed BIC charge `0.5 log n`;
4. cap state-local accepted swaps at `max(2, round(0.12*A)) + 1`, matching the positive-family perturbation bound plus one slack move;
5. alternate shared-key and local-state coordinate sweeps three times;
6. candidate mode/period selection uses final language score minus the same local-swap BIC charge and a fixed `0.5*(period-1)*log n` schedule charge.

No circular ordering of symbols is used.

## Fresh development instances

v0.2 diagnostics and development use synthetic replicate IDs offset by `+1000` relative to v0.1. This produces fresh codebooks/state maps and avoids selecting the new search algorithm on the exact v0.1 smoke instances. Locked test remains the untouched `test` split with a separate `+2000` replicate offset.

## Qualification sequence

Before hidden segmentation is reconsidered:

### A. True-structure key gate

Supply true unit heads, true mode and true period; hide plaintext and all keys.

On 8 fresh German development trials at length 192:
- mean recovery >= 0.90;
- median >= 0.95;
- at least 7/8 >= 0.85.

If this fails, v0.2 stops. Segmentation is not relevant yet.

### B. Blind-structure head gate

True heads only; hide mode, period and keys. Same 8 fresh trials:
- mean recovery >= 0.85;
- median >= 0.90;
- >=7/8 recovery >=0.80;
- mode accuracy >=7/8;
- period accuracy >=6/8.

### C. Segmentation replacement

Only after A and B pass. The v0.1 transition-surprisal boundary lattice is already known to be inadequate (~0.676 smoke F1) and is not promoted to a binding v0.2 gate. v0.2 must freeze a new information-preserving joint segmentation method before testing it.

### D. Joint and locked gates

The original v0.1 Gate-2/Gate-3 recovery, boundary-F1, structure and hostile-control thresholds remain minima. A new locked `test` split with `+2000` replicate offset must pass before any Voynich runner is authorised.

## Target seal

Voynich remains sealed. v0.2 contains no Voynich loader. No target-runner change is permitted before the synthetic locked gate.