# SVT v0.4 — Hidden segmentation qualification

Status: FROZEN BEFORE BINDING EXECUTION
Date: 2026-08-21
Binding execution: PR #21

## Motivation
SVT v0.3.4 passed blind state/key recovery with autonomous primitive-period canonicalisation. The remaining unqualified layer is hidden 1–3-glyph unit segmentation. The original v0.1 local-transition-surprisal segmenter is CLOSED as inadequate (about 0.676 boundary F1 in smoke tests) and is not reused as the primary method.

Voynich remains sealed. This stage uses synthetic known-answer data only.

## Mechanism held fixed
The renderer remains the frozen FSVT mechanism:
- each plaintext/head unit produces 1, 2, or 3 visible glyphs;
- the first visible glyph is the stateful substitution head;
- continuation glyphs are generated from within-unit modular offsets with the frozen continuation noise;
- no separators identify unit boundaries.

No state/key or renderer parameter is altered in v0.4.

## New segmentation model
v0.4 introduces one new inference component only: a ciphertext-only semi-Markov segmentation model.

For a proposed unit of length L in {1,2,3}, the model scores:
1. the frozen global length prior (0.30, 0.45, 0.25);
2. a learned head-symbol categorical distribution;
3. for L>=2, a learned categorical distribution over the modular difference from head to continuation-1;
4. for L=3, a learned categorical distribution over the modular difference from continuation-1 to continuation-2.

All categorical distributions use symmetric Dirichlet smoothing alpha=0.5. The model is deliberately state-, language-, and plaintext-blind. It sees only the rendered glyph sequence and known line starts.

Inference alternates:
- Viterbi semi-Markov segmentation of each line under the current emissions;
- maximum-a-posteriori re-estimation of head and continuation-difference emissions.

Exactly 12 EM iterations and 6 deterministic initialisations are used:
- one legacy-surprisal path only as an INITIALISATION (never as final scoring truth);
- one near-length-2 deterministic tiling;
- four frozen-seed random tilings drawn from the fixed length prior.

The selected segmentation is the restart with highest final ciphertext-only model score. Plaintext truth and true boundaries are revealed only after selection.

## Binding Gate S0 — segmentation component
- Language source used to generate synthetic plaintext: German (`de`), but the segmenter receives no language model.
- Split: `dev`.
- Plaintext/head length: 1536 units.
- Fresh replicate namespace: offset 17000.
- Eight trials: generator mode in `{periodic,line_reset}` x replicate `0..3`.
- These replicates are disjoint from v0.3–v0.3.4 and all earlier segmentation diagnostics.

Metrics:
- boundary F1 against true hidden head positions;
- inferred-vs-true unit count ratio;
- legacy local-surprisal boundary F1 recorded as a nonbinding comparator.

PASS requires all:
- mean boundary F1 >= 0.90;
- median boundary F1 >= 0.90;
- minimum boundary F1 >= 0.85;
- 8/8 trials boundary F1 >= 0.85;
- mean absolute unit-count relative error <= 0.05.

Failure closes this state-blind semi-Markov segmenter as the primary v0.4 boundary method. It does not authorize target use or threshold tuning. A failure may motivate a separately versioned state-aware/joint segmenter.

## Next stage if S0 passes
Build v0.4.1 full end-to-end gate on a new untouched namespace:
`surface -> hidden segmentation -> blind mode/primitive period/key -> plaintext`, using the already-qualified v0.3.4 state/key solver unchanged.

The end-to-end gate must include both boundary and plaintext recovery and hostile controls before any Voynich run.

## Target seal
This runner contains no Voynich loader. `voynich_opened=false` throughout.