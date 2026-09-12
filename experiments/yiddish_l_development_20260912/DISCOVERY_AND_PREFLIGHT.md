# Yiddish L development feasibility — discovery/preflight freeze

Date: 2026-09-12
Parent: yiddish qualification v03 (`f29c7befd809f2db9af68ea9d8c42b25bcb06f5b`)

## Scope

This branch is DEVELOPMENT / corpus-engineering only. It cannot issue an L certificate and must not load or score the Voynich target.

The controlling parent result remains: finite-panel 1501–1600 short-message R pass in normalized Penn representation; long historical cell corpus-limited; primary diplomatic representation unqualified; L/T unqualified.

## Ordered work

1. Search for source-disjoint historical Yiddish beyond PPCHY, prioritising the DFG/LAUDATIO `Historische Syntax des Jiddischen` corpus (~10 texts / ~42k tokens) before creating any new transcription.
2. Acquire a reproducible Early New High German comparison pool from ReF 1350–1650, prioritising southern varieties and matched period/register where support exists.
3. Before any language-score fitting, write and hash a source manifest assigning whole relationship groups to BUILD_SOLVER / BUILD_EVALUATOR / DEVELOPMENT. Previously consumed Yiddish sources may be used only for development.
4. Run a Yiddish-vs-German DEVELOPMENT feasibility benchmark with equal admissible token budgets. Search objective and evaluator data/models remain separated as required by protocol v1. The initial purpose is falsification: determine whether Yiddish can be distinguished from the closest registered language under the recovered M0 pipeline without relying on metadata or editorial artefacts.
5. Only if that development benchmark is viable, engineer the Hebrew comparator and a genuinely fresh Yiddish confirmation panel. No confirmation outcomes are to be unsealed before the complete L calibration/margin/abstention rule is frozen.

## Anti-leakage / anti-confound rules

- No Voynich transcription or target-derived statistic is read by the experiment.
- No source family may cross evaluator/development partitions.
- Chunks or keys from one work never count as independent works.
- Exact and near-duplicate passage checks precede fitting.
- Equal language training-token budgets use a deterministic work-balanced sampling rule fixed in the next source-manifest commit.
- Preserve source characters and offsets; any secondary normalization is a separately named representation.
- Explicitly test an editorial-representation nuisance baseline before interpreting language separation.
- A development positive is not L qualification; report DEVELOPMENT_FEASIBLE only.
- If historical source support is inadequate, report CORPUS_LIMITED rather than substituting modern German, biblical Hebrew, synthetic prose, or unreviewed OCR.

## Decision discipline

Before scoring, the executable manifest will fix source hashes, atomization, objective/evaluator algorithms, smoothing, budgets, seeds, calibration rule, pairwise margin/abstention rule, and nuisance tests. No outcome-driven source replacement is permitted.
