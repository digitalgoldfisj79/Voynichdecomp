# U2 mapping amendment v0.1.1 — single bounded repair

Date: 2026-08-14
Status: FROZEN BEFORE MODERN U2 TARGET CALCULATION

## Why this amendment exists

The original U2 mapping instrument compared two *linear* Currier-page-to-folio hypotheses (`skip12` and `keep12`) against 28 herbal A/B labels. It returned 23/28 and 15/28 respectively, below the preregistered PASS threshold of 26/28. The original mapping gate therefore remains **ABSTAIN_UNRESOLVED**; it is not retroactively reclassified.

Subsequent provenance work recovered an independent page↔folio concordance derived from Currier/Stolfi manuscript pagination rather than from the five anomaly outcomes. The programme-wide configuration permits one bounded repair. This amendment invokes it solely to replace the failed *linear inference instrument* with a direct concordance lookup.

## Anti-contamination statement

The direct concordance was recovered and frozen before calculating any modern replication status for pages 59, 60, 76, 79 or 94. None of the U2 clustering thresholds, feature definitions, historical anomaly pages, consensus rule, sample lengths or formal verdict boundaries are changed.

## Fixed historical anomaly mapping

- Currier p59 → f31r
- Currier p60 → f31v
- Currier p76 → f39v
- Currier p79 → f41r
- Currier p94 → f48v

## Fixed 40-page panel mapping

Biological block: p147=f75r, 148=f75v, 149=f76r, 150=f76v, 151=f77r, 152=f77v, 153=f78r, 154=f78v, 155=f79r, 156=f79v, 157=f80r, 158=f80v.

Herbal block: before the missing f12, Currier pagination follows successive recto/verso sides. After p22 the physically absent f12 is skipped, so p23=f13r and pagination proceeds successively thereafter. This rule yields the mappings used in the frozen panel, including the five explicit anomaly mappings above.

## Source basis

1. M. E. D'Imperio, *An Application of Cluster Analysis to the Question of “Hands” and “Languages” in the Voynich Manuscript*, NSA PI Informal No. 3, June 1978, Figure 6 (the 40 Currier page numbers and classes).
2. Jorge Stolfi Voynich archive, `fnum-to-pnum.tbl`, a page/folio concordance independent of U2 outcomes.
3. The programme's current Currier-language metadata is used only as a concordance cross-check, not to choose among clustering outcomes.

## Formal state

- original_linear_mapping_gate: `ABSTAIN_UNRESOLVED`
- bounded_repair_invoked: `YES`
- amended_mapping_gate: `PASS_DIRECT_CONCORDANCE`
- target_opened_at_time_of_freeze: `NO`
- remaining bounded repairs available: `0`

The U2 target may open only using this frozen direct concordance and the already-frozen U2 analysis code.