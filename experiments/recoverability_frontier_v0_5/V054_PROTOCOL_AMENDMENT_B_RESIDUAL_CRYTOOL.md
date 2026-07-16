# Recoverability frontier v0.5.4 — protocol amendment B: exhaustive residual-key search

Date: 2026-07-16

Status: fixed before execution.

## Evidence

The Stage A2 residual-character-key objective is capable of essentially complete recovery, but random-pair annealing is unreliable:

- `300,000 × 35`: five near-complete basin hits, mean 65.91%;
- `700,000 × 50`: seven near-complete hits, mean 88.41%;
- `1,200,000 × 70`: five near-complete hits plus one partial, mean 73.01%.

Increasing random-search duration does not produce monotonic reliability.

## Frozen replacement

Use the independently sourced CrypTool-style architecture already validated in v0.5.3:

- exhaustive deterministic sweeps over every pair of observed character-symbol assignments;
- linear cooling;
- calibrated initial acceptance target 0.05;
- independent restarts;
- exact preservation of the oracle-supplied observed plaintext-character inventory;
- fixed codeword plaintext spans are represented by locked key entries and never swapped;
- no inventory mutation.

The train-only quadgram objective and the eight Stage A2 development ciphertexts remain unchanged.

## Prefix curve

One run records the best objective result after:

`12, 24, 48 and 96` independent restarts,

with `3,000,000` pair proposals per restart. The minimum prefix passing the A2 gate is selected.

## A2 gate

- mean expanded character recovery at least 90%;
- median at least 99%;
- all eight development chunks at least 90%.

Failure blocks Stage B regardless of the A1 result.
