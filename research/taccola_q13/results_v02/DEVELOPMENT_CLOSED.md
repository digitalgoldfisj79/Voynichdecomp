# Taccola × Q13 v0.2 visual-instrument development — CLOSED

## RETRACTED FINDINGS

- **RETRACTED:** the raw Siena/Taccola concentration is not independent evidence. Family-collapsed Fisher p=0.318; the metric does not resolve this.

## v0.1 result

The frozen whole-page instrument failed calibration. Only page geometry passed; HOG and chamfer did not. Q13 remained sealed.

## v0.2 development round 1

Development used only exposed v0.1b feature artifacts. Neither Q13 nor the locked holdouts (BSB Clm 28800; BnF Lat.7239) was visually accessed.

Five fixed local representations were tested: Fourier contour, Hu moments, rotation-invariant polar mass, topology/skeleton, and local HOG. No local representation met the predeclared transfer rule.

## v0.2 development round 2

One final principled composite was tested: equal-block standardized Hu moments + skeleton/topology. Fixed motif-match sensitivity k=12/20/30 was evaluated.

At k=20:

- Clm197 → Palatino 766: z=1.516, p=0.0857, overall rank 3/35, technical rank 3/7, BSB uplift +0.41 SD.
- Palatino 766 → Clm197: z=1.507, p=0.0286, overall rank 1/35, technical rank 1/7, BSB uplift +0.30 SD.

The frozen development rule required technical rank ≤2 in both directions. It therefore **FAILED**. In the failing direction, Fontana Cod.icon.242 (0.846693) and Valturio (0.833475) both exceeded Palatino 766 (0.822372).

Sensitivity remained directionally stable, but this does not override the failed gate.

## Decision

- v0.2 freeze eligible: **FALSE**.
- Clm 28800 visual holdout remains **UNOPENED**.
- BnF Lat.7239 visual holdout remains **UNOPENED**.
- Q13 remains **SEALED**.
- No further visual-descriptor tuning is permitted on this branch. Further iterations would be post-hoc model selection against the same two autograph positives.

The visual programme therefore does not currently discriminate a Taccola-specific signature from the broader contemporary technical-manuscript visual class strongly enough to justify testing Voynich Q13.

Next research axis: codicological/production-mode evidence, treated independently of the failed visual instrument.
