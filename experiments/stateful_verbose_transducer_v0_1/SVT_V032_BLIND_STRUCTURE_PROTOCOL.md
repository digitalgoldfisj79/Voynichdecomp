# SVT v0.3.2 — Blind state-structure qualification

Status: FROZEN BEFORE EXECUTION

## Purpose
Qualify the already-passed SVT v0.3.1 multistart key solver when both state schedule class and period are hidden.

## Cipher family
Unchanged from SVT v0.3.1: shared global substitution key plus sparse state-local swaps; state schedule is either `periodic` or `line_reset`; true period is drawn by the frozen v0.1 generator. Variable-length surface segmentation is NOT opened at this stage. The head stream is supplied exactly so this stage isolates state-structure identification.

## Fresh data
- Language: German calibration corpus (`de`) only.
- Plaintext length: 1536 characters.
- Split: `dev`.
- Fresh replicate namespace: offset 9000.
- Eight target trials: 2 true modes × replicates 0..3.
- These trials are disjoint from v0.3, v0.3.1 and the outlier diagnostic namespaces.

## Hidden variables
For every trial the decoder is NOT given:
- true schedule class (`periodic` vs `line_reset`);
- true period;
- global substitution key;
- state-local perturbations.

The decoder is given only the exact latent head stream, line starts, candidate language model, and the frozen candidate family.

## Candidate space
- Mode: `periodic`, `line_reset`.
- Period: every integer 2..12.
- 22 structures per trial.

For each candidate structure run exactly 12 independent key-search starts. Within a candidate select the start with maximum frozen penalised model score. Across the 22 structures select the candidate with maximum frozen penalised model score. Plaintext truth is not consulted until after this selection.

No circular cipher alphabet, true-period hint, true-mode hint or plaintext-derived ranking is permitted.

## Binding Gate A3.2
All conditions must hold on the eight fresh trials:
1. exact `(mode, period)` recovery = 8/8;
2. mean plaintext recovery >= 0.95;
3. median plaintext recovery >= 0.97;
4. minimum plaintext recovery >= 0.85;
5. all 8 selected plaintext recoveries >= 0.90.

Failure of any condition closes v0.3.2. No Voynich target may be opened from a failed gate.

## Next stage if PASS
Build a separately frozen v0.4 hidden-segmentation gate. It must infer 1–3-glyph code boundaries jointly with the already-qualified state/key model. The Voynich arm remains sealed until that stage also passes known-answer calibration.

## Target seal
This protocol and its runners contain no Voynich loader. `voynich_opened=false` is binding throughout v0.3.2.
