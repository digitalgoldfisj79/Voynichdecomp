# PIII-CLOSURE results

## Formal verdict

**FAIL_TESTED_PIII_POLICIES**

The unchanged external *Polygraphia* III table was tested under the three published Hermes random-row policies (`iid10`, `iid24`, `iid131`) and two conservative line-aware policies (`cycle131`, `line_fixed`). All five primary arms were perfectly or effectively perfectly separable from the Voynich manuscript.

| Policy | Median grouped C2ST AUC | Median absolute critical z | Maximum absolute critical z | Compatible |
|---|---:|---:|---:|:---:|
| iid10 | 1.000 | 6.79 | 269.25 | No |
| iid24 | 1.000 | 6.49 | 107.60 | No |
| iid131 | 1.000 | 9.57 | 41.31 | No |
| cycle131 | 1.000 | 10.06 | 64.23 | No |
| line_fixed | 1.000 | 12.08 | 78.66 | No |

## Calibration

- random-label median AUC: **0.530** (required ≤0.60);
- within-line word-shuffle median AUC: **0.902** (required ≥0.80);
- within-token character-shuffle median AUC: **0.997** (required ≥0.80);
- policy-identification macro AUC: **0.998** (required ≥0.80).

Calibration passed. The classifier did not manufacture separation from random labels and reliably detected known sequence/morphology destruction.

## Scope

Each candidate was forced into the real Voynich folio and line token-count template. Four external plaintexts were used: Melanchthon, Latin *Secreta Secretorum*, Latin *Picatrix*, and Old Italian *Rettorica*. No Voynich-to-Polygraphia dictionary was fitted, the historical table was not changed, and Davis hand labels were not used.

The result closes the exact PIII table used at one codeword per plaintext character under the five primary policies. It does not close a newly invented or Voynich-fitted morpho-local nomenclator, changing scribe-specific codebooks, an additional surface-realisation layer, alternative segmentation, or sparse payloads.

Three complete CPU-performance executions returned the identical aggregate output. Exact physical bifolia were unavailable; whole approximate quire groups were used. The full signed compact results bundle is retained in the associated research handoff.