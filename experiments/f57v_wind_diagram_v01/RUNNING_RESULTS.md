# RUNNING RESULTS — f57v Wind-Diagram Comparator v0.1

## 2026-08-09 build checkpoint

Status: BUILT, NOT YET SCORED.

Completed:
- branch created from `main`;
- preregistration frozen before target scoring;
- visual-only feature codebook frozen;
- matched-within-manuscript analysis runner implemented;
- exact sign-test endpoint implemented;
- geometry / anthropomorph / breath / centre ablations implemented;
- metadata-blinding packet builder implemented;
- supplied f57v and BnF screenshot hashes recorded;
- seed source manifest created from authoritative catalogue/scholarly records.

Smoke test of the analysis runner succeeded locally on synthetic data. No empirical f57v score has been generated from those synthetic data and no result is inferred from them.

Key sourcing discovery during build:
- Oxford, Bodleian Library MS. Bodl. 646 is dated c.1460 and made at Padua. Scholarly catalogue literature describes fol. 34r as a table with twelve blowing heads of the winds. The same manuscript also supplies a useful hard-negative circular head/medallion comparator. This makes it a high-value matched stratum for the late-medieval northern-Italian sensitivity analysis.

Current blocker before legitimate execution:
- acquire the minimum preregistered image panel and same-manuscript controls;
- resolve the exact BnF shelfmark/folio/ARK for the user-supplied Exhibit B or leave it quarantined;
- generate blind packet and code visual features without metadata.

Do not inspect f57v against the coded corpus or tune thresholds before the acquisition/coding freeze is complete.
