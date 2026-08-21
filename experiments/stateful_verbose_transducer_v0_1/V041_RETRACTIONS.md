# Retracted / superseded findings

1. **RETRACTED:** “SVT v0.4 hidden segmentation failed.”
   - Reason: all eight S0 trial jobs and the aggregate computation succeeded; GitHub Actions failed later while uploading the aggregate artifact, so enforcement was skipped.
   - Correct binding S0 result reconstructed from immutable trial artifacts: PASS; mean F1 0.9491351144, median 0.9524384497, minimum 0.9219009638, 8/8 >=0.85, mean unit-count error 0.0266113281.

2. **REMAINS FAILED:** SVT v0.3.3 exact blind structure gate.
   - The 7/8 result is not retroactively promoted. Its harmonic alias failure motivated the separately frozen v0.3.4 primitive-period gate.

3. **SUPERSEDED BY SEPARATE PASSING TEST:** v0.3.4 demonstrated 8/8 exact ordinary primitive structure and 12/12 targeted harmonic primitive-period recovery. This does not rewrite the v0.3.3 ledger.

Voynich remains sealed for the SVT programme.
