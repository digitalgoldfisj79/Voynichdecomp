# AMENDMENT 002 — Folio/section post-hoc audit

Date: 2026-08-15

## Status

This is a **post-hoc audit/implementation repair**, not a change to the preregistered primary analysis. The primary v0.2 verdict `NO_ROBUST_MAGIC_AFFINITY` is frozen and remains unchanged.

## Trigger

The completed v0.2 result showed an aggregation inversion: all four whole-corpus representations were closer to class A (ordinary medieval prose), while a majority of individual folios had positive `delta_magic`. The planned section analysis reported `NOT_PARSEABLE`.

Inspection showed that `voynich_section_map.json` stores the folio map under the top-level key `mapping`, while the original `voy_section_F5` normalizer only inspected top-level values. This prevented section grouping.

## Allowed repair

1. Read section labels from `section_map["mapping"]` when present; otherwise preserve the original flat-map behavior.
2. Recompute the exact frozen external calibration deterministically, without changing corpus splits, feature definitions, FDR qualification, generators, thresholds, or target representations.
3. After external freeze reproduction, acquire the same pinned RF / bitrans / STA-aaa sources and create the same four target layers.
4. Export per-folio distances to A/B/C and `delta_magic` in the exact qualified metric space.
5. Aggregate folio scores descriptively by section and also score each section as concatenated text.
6. Export A-vs-C (`delta_C`) separately because the frozen B class has only one held-out test block and B-distance is therefore unstable.
7. Diagnose token-count dependence of folio scores. No new primary p-value or success threshold is introduced.

## Interpretation constraint

These outputs are secondary/descriptive. They may identify where the corpus-level inversion arises and motivate a future preregistered test, but they cannot overturn the frozen primary verdict by themselves.
