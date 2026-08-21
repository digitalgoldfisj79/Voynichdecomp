# SVT v0.3.3 — Calibrated shortlist blind state-structure gate

Status: FROZEN BEFORE BINDING EXECUTION

## Motivation
The exhaustive v0.3.2 176-cell implementation is statistically valid but GitHub runner allocation serialises the matrix. This v0.3.3 preserves the scientific question while using a truth-free screen calibrated only on previously used synthetic namespaces.

## Screening calibration already completed
On 16 non-binding trials from offsets 5000 and 7000, the cheap statewise-frequency schedule screen ranked the exact true `(mode, period)` within the top 6 in 16/16 cases; median truth rank 1; maximum rank 6. No binding namespace was used to choose K.

## Binding data
- Language: German (`de`).
- Length: 1536 plaintext characters.
- Split: `dev`.
- Fresh replicate namespace: offset 11000.
- Eight trials: true mode in {periodic, line_reset} × replicate 0..3.
- This namespace has not been used by v0.3, v0.3.1, v0.3.2, or the screen diagnostic.

## Hidden variables
For every binding trial the decoder is not given true mode, true period, global key, or state-local perturbations. The exact latent head stream is supplied; hidden variable-length segmentation remains reserved for the next stage.

## Stage 1: truth-free screen
Evaluate all 22 candidate structures (2 modes × periods 2..12) using only independent statewise frequency-key initialization and the candidate language model. Apply the frozen sparse schedule penalty. Retain exactly the top 6 candidate structures. Plaintext truth is not consulted.

## Stage 2: qualified key solver
For each of the six retained structures run exactly 12 independent v0.3.1 key starts. Select the best start by frozen penalised model score. Then select the best of the six structures by the same penalised model score. Plaintext truth is revealed only after the final selection.

## Binding Gate A3.3
All must hold across the eight fresh trials:
1. the true structure is present in the screen top 6 for 8/8 trials;
2. exact selected `(mode, period)` recovery = 8/8;
3. mean plaintext recovery >= 0.95;
4. median plaintext recovery >= 0.97;
5. minimum plaintext recovery >= 0.85;
6. all 8 plaintext recoveries >= 0.90.

Failure closes v0.3.3. No target reinterpretation or gate relaxation is allowed.

## Next stage if PASS
Build a separately frozen hidden-segmentation v0.4 in which 1–3-glyph boundaries are latent and are solved jointly with this qualified state/key machinery.

## Target seal
The v0.3.3 runner contains no Voynich loader. `voynich_opened=false` throughout.
