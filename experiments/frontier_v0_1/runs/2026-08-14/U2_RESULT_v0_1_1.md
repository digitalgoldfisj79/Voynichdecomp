# U2 — D'Imperio 1978 anomaly replication, v0.1.1

Date: 2026-08-14
Formal verdict: **ABSTAIN_UNRESOLVED**
Target opened: YES, only after `U2_MAPPING_AMENDMENT_v0_1_1.md` was committed.

## Provenance correction

The earlier NSA-audit ledger described Figure 6 as 12 Biological-B / 11 Herbal-A / 17 Herbal-B. Re-reading the primary 1978 Figure 6 and D'Imperio's narrative shows this extraction was wrong. The frozen 40-page panel is **12 Biological-B / 16 Herbal-A / 12 Herbal-B**. The 12 Herbal-B pages are consistent with D'Imperio's own narrative: a core seven-page Herbal-B cluster plus the five repeatedly anomalous Herbal-B pages 59, 60, 76, 79 and 94.

The original *linear* mapping gate remains ABSTAIN (skip12 23/28; keep12 15/28, threshold 26/28). The programme's single bounded repair replaced that failed inference instrument with the independently recovered direct Currier/Stolfi page↔folio concordance before this target calculation.

Historical anomaly mappings: p59=f31r; p60=f31v; p76=f39v; p79=f41r; p94=f48v.

## Frozen analysis

The existing `dimperio_replication.py` was followed exactly for the primary monographic arm:

1. leave-one-out nearest class centroid under correlation distance;
2. Ledoit-Wolf within-class Mahalanobis outlier, 95th percentile;
3. average-linkage correlation clustering forced to three clusters, then majority-class mismatch.

A historical anomaly counts as replicated when at least two of the three instruments flag it. Frozen decision: ≥4/5 CONFIRM; ≤1/5 FALSIFY; 2–3 ABSTAIN.

Primary historical representation: Prescott Currier PCCA/PCCI (identical on this panel). Input `voynich_transcriptions_slim.json` SHA-256: `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`.

## Results

| representation | sample | replicated historical pages | n | arm verdict |
|---|---:|---|---:|---|
| **PCCA** | **350** | **59, 60, 76** | **3/5** | **ABSTAIN** |
| **PCCA** | **400** | **59, 60, 76** | **3/5** | **ABSTAIN** |
| **PCCA** | **full** | **59, 60, 76** | **3/5** | **ABSTAIN** |
| ZLZI | 350 | 59, 60, 76, 79 | 4/5 | CONFIRM in this arm only |
| ZLZI | 400 | 59, 60, 76 | 3/5 | ABSTAIN |
| ZLZI | full | 59, 60, 79 | 3/5 | ABSTAIN |
| TTII | 350 | 59, 60, 76 | 3/5 | ABSTAIN |
| TTII | 400 | 59, 60, 76 | 3/5 | ABSTAIN |
| TTII | full | 59, 60, 79 | 3/5 | ABSTAIN |
| TTVE | 350 | 59, 60, 76 | 3/5 | ABSTAIN |
| TTVE | 400 | 59, 60, 76 | 3/5 | ABSTAIN |
| TTVE | full | 59, 60, 79 | 3/5 | ABSTAIN |
| VDRB-1 | 350 | 59, 60, 76 | 3/5 | ABSTAIN |
| VDRB-1 | 400 | 59, 60, 76 | 3/5 | ABSTAIN |
| VDRB-1 | full | 59, 60, 79 | 3/5 | ABSTAIN |
| GCGI | 350 | 59, 60, 76 | 3/5 | ABSTAIN |
| GCGI | 400 | 59, 60, 76 | 3/5 | ABSTAIN |
| GCGI | full | 59, 60, 76 | 3/5 | ABSTAIN |

The isolated ZLZI-350 4/5 result is not stable to the preregistered 400-character or full-page sensitivities and is therefore not promoted over the primary PCCA result.

## Mandatory naive diagnostic

The older preregistration required the naive character partition `first_1, first_2, last_1, last_2, last_3`. It was operationalised as namespace-qualified positional strings per token, represented by page-normalised category frequencies, then passed through the same three instruments. Because that exact vectorisation was not encoded in Frontier v0.1 before the primary result, this is retained as a mandatory **descriptive diagnostic**, not a new confirmatory gate.

PCCA naive baseline:

- 350: 2/5 (59, 79)
- 400: 2/5 (76, 79)
- full: 1/5 (79)

Thus the naive partition does **not** reproduce the historical anomaly set as well as the monographic PCCA primary (3/5), but this does not rescue the primary from its preregistered ambiguous band.

## Adjudication

D'Imperio's five anomalous Herbal-B pages were not simply erased by modern transliteration: pages 59 and 60 are highly persistent, and page 76 is persistent at historical sample lengths. However, the frozen experiment does not recover ≥4/5 robustly. Page 94 is especially method-dependent: agglomerative clustering repeatedly treats it anomalously, but the independent centroid/outlier instruments do not give the second vote required by the consensus gate.

**Formal U2 result: ABSTAIN_UNRESOLVED.**

No historical/provenance inference is permitted from U2.