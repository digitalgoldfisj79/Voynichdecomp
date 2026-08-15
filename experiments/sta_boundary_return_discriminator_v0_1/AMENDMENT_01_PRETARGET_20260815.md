# Amendment 01 — synthetic calibration operationalisation

Date: 2026-08-15
Status: **PRE-TARGET**. No B0/B1/B2/B3 target output has been generated or inspected.

The main preregistration states that the template-positive synthetic control must identify B2 as sufficient and that the return-positive control must recover a planted q=0.08 and pass the predictive identity gate. Because the synthetic template source is not constructed to reproduce the Voynich-specific P1-P5 magnitudes, applying the Voynich target gates to CT/CR would be category error.

The frozen operationalisation is therefore:

- **CT template-positive:** 8 independent synthetic B2 corpora in the full run (2 in smoke mode). Median refitted boundary q must be <=0.02 and **none** of the synthetic template-only replicates may pass the B3 held-out predictive identity gate.
- **CR return-positive:** 8 independent matched synthetic corpora with planted boundary q=0.08. Median recovered q must be within +/-0.03 of 0.08 and at least 75% of replicates must pass the B3 held-out predictive identity gate.
- Predictive identity gates use the same folio-held-out logic and 2,000 folio-block bootstrap resamples as the target (100 only in smoke mode).

No target threshold, model, boundary definition, fold, feature, q range, or scientific adjudication changes.