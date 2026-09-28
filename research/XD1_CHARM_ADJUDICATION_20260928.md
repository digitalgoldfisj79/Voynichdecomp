# XD1-CHARM final adjudication — 2026-09-28

## Decision

**SGT12 is NOT downgraded by the medieval charm control.**

The frozen XD1-CHARM downgrade gate does not fire in either the primary 14-unit Add. 9308 charm cohort or the stricter 10-unit subset restricted to texts explicitly labelled/called charms in the manuscript. This follow-up was preregistered as downgrade-only. It does not retroactively strengthen the five original SGT12 promotion gates and does not constitute independent replication.

## Primary source and power

Cambridge, University Library, MS Add. 9308 (late 14th/early 15th c., England), independently included in Heather A. Taylor's 48-manuscript charm/experimenta survey and independently catalogued/transcribed by Cambridge's Curious Cures project. CUDL exposes 183 diplomatic transcription pages whose manuscript physical lines carry source transcription-line IDs and polygon geometry. No OCR-derived or browser-wrapped lines were used.

Fourteen charm units were frozen before any recurrence outcome. Whole-line boundary rule: mixed recipe/charm boundary lines were excluded, never split.

- Primary: 215 physical lines, 1,480 tokens, 1,050 lag-2 opportunities.
- Strict explicit subset: 154 physical lines, 1,058 tokens, 750 lag-2 opportunities.
- Frozen power floor (>=4 units, >=100 lag-2 opportunities): PASS.

## Primary 14-charm result

| Metric | lag 1 | lag 2 |
|---|---:|---:|
| observed | 0.0031621 | 0.0209524 |
| null mean | 0.0198779 | 0.0199143 |
| observed/null | **0.1591** | **1.0521** |
| effect | -0.0167158 | +0.0010381 |
| null SD | 0.0032585 | 0.0036168 |
| effect/null SD | **-5.13** | **+0.29** |

Lag 1 is strongly suppressed. For lag 2: **the metric does not resolve this.** The primary charm population therefore does not reproduce the VMS pattern of unresolved lag 1 plus resolved lag-2 enrichment.

## Strict ten-charm sensitivity

- lag1 observed/null 0.1204; effect/null SD -4.61.
- lag2 observed/null 1.3154; effect/null SD +1.44.

Lag 1 remains strongly suppressed. For lag 2: **the metric does not resolve this.** The larger lag-2 ratio is not resolved by the matched permutation null and fails the frozen joint gate.

## Robustness

All 14 leave-one-unit-out populations fail the downgrade gate. Across leave-one-out populations, lag1 remains about -4.72 to -5.37 SD and observed/null about 0.095 to 0.183. No leave-one-out population produces unsuppressed lag1 plus resolved enriched lag2.

The Tres boni fratres wound charm is individually lag-2 enriched (observed/null 2.172; +2.28 SD), but lag1 is simultaneously strongly suppressed (observed/null 0; -2.21 SD). It therefore fails the joint SGT12 geometry. This is why lag-2 enrichment alone was explicitly retracted as diagnostic after the ReM control.

## Length sensitivity

The only formal frozen stratum with >=50 charm lines is 6–10 tokens: n=180; lag1 observed/null 0.101, -4.83 SD; lag2 observed/null 1.024, +0.11 SD. The pooled result is therefore not an artifact of mixing very short and long lines.

The 2–5 token arm has only 32 lines and is non-formal: lag1 -1.23 SD, observed/null 0.548; lag2 +1.86 SD, observed/null 1.852. Neither reaches the preregistered 2-SD resolution rule.

## Unit-cluster bootstrap

Resampling the 14 charm units gives lag1 effect 95% interval [-0.02133, -0.01158], entirely negative; lag2 [-0.00801, +0.00938], spanning zero.

## Same-manuscript medical background

Add. 9308 medical background (descriptive only, not certified pure recipes): 2,596 physical lines, 19,768 tokens. Lag1 observed/null 0.0122, -21.19 SD. Lag2 observed/null 0.7721, -4.59 SD.

Preregistered cluster bootstrap of charm-minus-background effect: lag1 +0.002179 / bootstrap SD 0.002614 = +0.83 SD; lag2 +0.005459 / 0.004740 = +1.15 SD. **The metric does not resolve either charm-vs-background difference.**

## VMS contrast

Frozen VMS SGT12: lag1 observed/null 1.024, +0.47 SD — **the metric does not resolve this**; lag2 observed/null 1.224, +4.16 SD. Primary Add. 9308 charms: lag1 0.159, -5.13 SD; lag2 1.052, +0.29 SD.

The tested charms therefore do not provide the obvious historical-register explanation for the VMS joint recurrence geometry.

## German sensitivity

Voynich Ninja thread 6047 supplies Amberg Ms. 77 as a German medicinal charm witness, but that selection is Voynich-aware and was preregistered as sensitivity-only. The BSB/MDZ route supplies images and potentially OCR/hOCR, not an independently supplied diplomatic physical-line transcription comparable to Add. 9308. Under the frozen lineation rule, OCR/reconstructed lines are inadmissible.

Status: **BLOCKED_PHYSICAL_LINEATION. No German recurrence outcome computed.**

## Corrections and engineering ledger

1. First CUDL preflight falsely treated absence of TEI lb/line tags as absence of physical lineation. Source-only inspection showed CUDL diplomatic HTML encodes each manuscript line with transcription-line ID and polygon geometry. Corrected before any recurrence outcome.
2. Initial post-outcome secondary-bootstrap workflow failed on a Python import path before computing statistics. Import repaired without changing selection/statistic; successful rerun produced the reported secondary result.
3. No charm result from the old alleged 13-Aug run is reinstated. That historical claim remains RETRACTED/UNSUPPORTED.

## Reproducibility

Branch: supergrammar-xd1-charm-20260928

Frozen protocol files: research/PROTOCOL_XD1_CHARM_20260928.md; research/PROTOCOL_XD1_CHARM_v11_ADDENDUM_20260928.md; research/XD1_CHARM_MANIFEST_20260928.json.

Primary outcome: GitHub Actions run 36470089475; artifact xd1-charm-outcome ID 10990694141; artifact SHA256 50c46032fe328e2012437e06550d6446c92cb26a62c9781bb893b00cadbcf316; result JSON SHA256 49950bb4b2b386bcaaf75e67a679d8c85d03c24ee115b72121067ecc95cd0a3e; summary TSV SHA256 92d1cb4da11670170f41d7909ff72d058f3f0e25eb7b97e3bdc0a746bea8da9f.

Secondary bootstrap: successful run 36470832357; artifact ID 10991605806; artifact SHA256 b23d8b761bae3569d6e436b35723b02f2b6e638cfb3392698348831daeb826c1; result SHA256 cb5e247e512f92e52078f16e610b317795fa1a69abd717362681773939ff4776.

## Licensed update

The SGT12 claim remains bounded. A substantial independently selected late-medieval English medical-charm control now fails to reproduce its joint recurrence geometry: the charm cohort strongly suppresses immediate exact-token repetition and does not resolve lag-2 enrichment. This closes the previously open charm-control gap for the tested English corpus, but it does not establish universality across charm traditions, languages, regions or transmission contexts. A German diplomatic-line sensitivity remains open.