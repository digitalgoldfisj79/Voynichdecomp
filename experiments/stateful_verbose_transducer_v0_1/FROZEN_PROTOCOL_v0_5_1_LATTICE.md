# SVT v0.5.1 — Frozen Lattice Coupling Protocol

Date frozen: 2026-08-21

## Purpose

Final qualification attempt for the stateful verbose transducer (SVT) mechanism class.

The independently qualified components are:

1. state/mode/primitive-period/key recovery when cipher-unit boundaries are supplied (SVT v0.3.4; German and Latin portability confirmed by v0.5.0);
2. standalone hidden 1–3-glyph segmentation from surface evidence (SVT v0.4; German and Latin portability confirmed by v0.5.0).

v0.4.1 failed because hard segmentation errors desynchronised the state clock. v0.4.2 and v0.4.3 failed because a single joint objective respectively under- and over-segmented. v0.5.1 therefore does not use a blended joint objective.

Voynich remains sealed. No Voynich data, metrics, glyphs, text, labels, or target-derived tuning may enter development or binding.

## Frozen architecture

### A. Surface-owned lattice

Fit the unchanged v0.4 semi-Markov surface model.

Using its fitted emissions, enumerate the top **K = 8** complete segmentation paths. The surface model alone determines which paths enter the lattice.

### B. Length-matched cipher evidence

For each path:

- extract the head glyph stream;
- preserve its inferred head count and per-line head counts;
- compute the best cheap blind state-structure screen over the unchanged candidate set:
  - mode ∈ {periodic, line_reset};
  - period ∈ {2,...,12};
- generate **8** matched null head streams by shuffling heads independently within inferred lines;
- recompute the same best blind screen for each null;
- define cipher evidence as

  z = (actual best screen - matched-null mean) / matched-null standard deviation.

Because every null has exactly the same inferred number of heads and the same inferred line-head counts as its candidate, changing segmentation length has no direct scoring advantage.

### C. Reranking rule

- If the largest candidate z-score is at least **2.0**, select that candidate.
- Otherwise retain the surface-MAP candidate (surface rank 0).
- No weighted surface/language mixture is permitted.
- No truth-derived unit-count penalty is permitted.

### D. Full key recovery

Run the unchanged v0.3.4-style solver only once, on the selected segmentation:

- cheap screen all 22 mode/period structures;
- refine top 6;
- 12 starts per refined structure;
- same factorised state-key optimiser and BIC penalties;
- primitive-period canonicalisation by proper divisors.

Truth is used only after selection for evaluation.

## Development calibration: already-spent material only

Exactly 8 trials:

- German: v0.4.2 namespace offset 23000, dev split;
- Latin: v0.5.0 segmentation namespace offset 33000, dev split;
- modes: periodic and line_reset;
- replicates: 0 and 1 for each language × mode.

Development PASS requires all:

- n = 8;
- mean selected boundary F1 ≥ 0.90;
- minimum selected boundary F1 ≥ 0.85;
- mean absolute unit-count error ≤ 0.05;
- |mean signed unit-count error| ≤ 0.03;
- mean |selected count - surface-MAP count| / MAP count ≤ 0.03;
- exact mode + primitive period ≥ 6/8;
- mean plaintext sequence recovery ≥ 0.85;
- minimum plaintext sequence recovery ≥ 0.70.

If development fails, the programme stops. No fresh binding namespace is opened.

## Final bilingual binding gate

Runs only if development passes.

Exactly 16 untouched synthetic trials:

- German and Latin;
- split = test;
- German namespace offset 37000;
- Latin namespace offset 39000;
- modes: periodic and line_reset;
- replicates: 0,1,2,3 for each language × mode.

Binding PASS requires:

- n = 16;
- exact mode + primitive period = 16/16 overall and 8/8 within each language;
- every boundary F1 ≥ 0.85;
- mean boundary F1 ≥ 0.90 within each language;
- mean absolute unit-count error ≤ 0.05 within each language;
- |mean signed unit-count error| ≤ 0.03 within each language;
- mean absolute selected-vs-surface-MAP count shift ≤ 0.03 within each language;
- every plaintext sequence recovery ≥ 0.85;
- mean and median plaintext sequence recovery ≥ 0.90 within each language.

## Stop rule

This is the final SVT coupling attempt.

- Binding PASS: SVT synthetic qualification is complete; proceed only then to hostile/wrong-language controls before any Voynich exposure.
- Development FAIL or binding FAIL: close the SVT mechanism as **not qualified for blind application to Voynich**. Do not create v0.5.2/v0.5.3 tuning iterations from these results.

No threshold or architecture change is permitted after opening the fresh binding namespace.
