# S56: Joint RF/ZL line-bootstrap chronology audit (18 October 2026)

## RETRACTIONS AND SUPERSESSIONS (READ FIRST)

**R7:** The earlier v2 proxy intersection implied an 18/32 split at 99% support. The actual full joint four-path refit **resolves no split at 99%**. Withdraw the earlier proxy statement as a result of full text+representation uncertainty.

**R6 remains controlling:** Fixed-MDS chronology overstates fine within-group certainty; the physical-adjacency-disagreement verdict was withdrawn; independent vocabulary information beyond glyph-transition style was not established; partial respelling was not excluded.

**Metadata correction:** RF and ZL use paired pseudorandom line-index selections within sheets (same number of lines per sheet), not independent draws across representations. Point masks are matched.

## Execution

[GitHub Actions run 39161321555](https://github.com/digitalgoldfisj79/Voynichdecomp/actions/runs/39161321555) completed all refit shards and aggregation successfully.

- **3,200 / 3,200 fits converged**: 100 bootstrap draws × ZL/RF × two penalties (0.01/0.05) × eight starting routes. All **200 checkpoint pickles** recovered and independently replayed.
- Both corpora reconstruct **4,130 paragraph lines**, zero error for word- and joined-line transition totals; original matrix distances recovered to machine precision (ZL 8.33e-17, RF 5.55e-17).
- Each draw resamples whole transcription lines within bifolia; re-estimates glyph-transition distances and MDS; then refits ElPiGraph on matched published 90%-inclusion masks.
- Retain converged trajectories with 1a before 1b. Paths 2–4 additionally require their chosen 3a/3b and 4a/4c group-median directions. Choose minimum objective per representation, penalty and replicate; aggregate pairwise same-fit after/half-tie frequencies with equal setting weights. Scores/ranks checked exactly by independent replay.
- S56 [Python refit source](scripts/s56_joint_refit_20261018.py) and [workflow](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/research/v2-joint-bootstrap-20261018/.github/workflows/voynich-s56-joint-bootstrap.yml). Frozen eo777 source main commit 97c8e24b468b69a09771476122f029b40d355ead; S56 workflow head 7c6731e3bf0e740d85b55ca2025157f708951f96.

## Result

**Criterion:** Every bifolium left of a boundary must precede every bifolium to its right with at least the threshold same-fit support across all four path scenarios.

| Threshold | Shared boundaries | Block sizes |
|---|---|---|
| 90% | 10, 18 | **10 / 8 / 32** |
| 95% | 10, 18 | **10 / 8 / 32** |
| 99% | none | **50** |

This is conditional model stability, not chronological probability. **No matched-null SD** exists for these descriptive block-count headlines.

Independent 95% partitions: Path 1 default **10/8/5/15/4/1/2/5**; Path 2 **10/8/5/27**; Path 3 **10/8/20/4/1/2/5**; Path 4 **10/8/20/5/7**.

Fine order: in Path 1 **28/126** within-group pairs are ≥95%-decisive versus **47/126** in the fixed-geometry model, a descriptive decline of **19 pairs (15.1 percentage points)**. No matched-null SD was estimated for this difference, so no inferential z is claimed. Just **20/126** within-group pair directions remain consistent and ≥95% in every refitted path. **140/1,225** pairs flip majority direction between scenarios (null SD not calculated).

The 1.5a-before-2a contrast is strong *conditional on the model*, but a universal block after 1.5a fails because units from other groups can intervene: Path 4 has preference only **0.880** for Q:93-96 preceding J:69-70. Later 3a/3b and 4a/4c orders reverse depending on conditioned path.

**Cautions:** 1a-before-1b was imposed in the trajectory acceptance rule. RF and ZL change transliteration and glyph unitisation together. No prospective physical anchor, calibrated known-order matched negative, or independent date evidence identifies historical production chronology.

## Audit and reproducibility

All fit checkpoints and the final scores/ranks remain GitHub Actions artifacts (90-day retention). The independent local audit reproduced ranks/scores exactly from the 200 bootstrap pickle files and recomputed cross-path thresholds.

Final disposition: **the detailed revised chronology is not resolved jointly; coarse partition 10/8/32 at 90–95%, none at 99%.** The older seven-stage ZL result remains *scenario-specific*, not universal.

Outstanding test: independently dated or original-state vellum/soiling anchors chosen prospectively; known-order manuscript controls matched for length, section and scribe.
