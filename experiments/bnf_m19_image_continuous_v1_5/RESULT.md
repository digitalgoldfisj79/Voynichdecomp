# BnF M19 Image Bridge v1.5 — Result

Protocol freeze: `321c2074ab6a84356efd20aa790de4fdc78ce284`
Primary qualification job: `6a7842c63e1f34a7e32c017f` (cancelled prospectively once the gate became impossible)
Strong-init control job: `6a7844853e1f34a7e32c01a1` (cancelled after Latin disproved the convergence explanation)
Exact-letter control job: `6a78456f3e1f34a7e32c01c0` (completed)

Verdict: **CONTINUOUS IMAGE INSTRUMENT NOT QUALIFIED**. No Voynich H12/C12 language score was generated.

## Image-only calibration

The frozen dense-segment boundary model used lambda=.12. Image-only Gaussian calibration selected:

- sigma0 = 0.0102182793 per 768-dimensional coordinate;
- selected scale = 1.0;
- selected sigma = 0.0102182793.

Tvis Gaussian-mixture log likelihoods selected this value before any language fit.

## Primary synthetic qualification failure

The v1.5 continuous 19-state approximation failed its known-plaintext controls. Results observed before prospective cancellation included:

- Latin: top Arabic, Latin rank 3, recovery 0.4027, fit agreement 0.6734;
- Italian: top Italian but margin 0.0453 (<0.03 language threshold is actually passed, but recovery 0.4407 and agreement 0.5770 fail);
- German: top German, margin 0.0986, recovery 0.7914, agreement 0.7410;
- French: top French by only 0.00151, recovery 0.6377, agreement 0.3885;
- Arabic: top Arabic, margin 0.1951, recovery 0.5828, agreement 0.6863.

The frozen qualification required 6/6 language recovery, median state recovery >=0.90, minimum >=0.80 and fit reproducibility >=0.90. Qualification was already impossible before Spanish completed, so the paid GPU job was cancelled and Voynich stayed sealed.

## Convergence diagnosis

A control-only rerun replaced the 6k x 3 hard-permutation initializer with the proven v0.9-scale 24k x 6 search. Latin still failed:

- top Arabic;
- Latin rank 3;
- state recovery 0.4461;
- independent-fit agreement 0.6259.

Therefore weak hard-map initialization does not explain v1.5's failure.

## Exact 23-letter diagnosis

A second control-only diagnostic removed the approximate 19-state language prior entirely. It used the exact generative structure:

`23-letter language HMM -> BnF letter/value mixture -> continuous Gaussian image emission`,

with exact forward/backward and exact posterior responsibilities over letter/value pairs.

The fresh Latin synthetic control still failed:

- top Arabic;
- Latin rank 3;
- Arabic-vs-runner-up visual-gain margin 0.1045;
- correct Latin state-mean recovery 0.4029;
- independent Latin fit agreement 0.5100.

Thus the failure is not caused by the 19-state Markov approximation. Under realistic overlap calibrated from the image data, the free 19 continuous emission means can fit multiple language priors well enough that the numerical-state identities are not recoverable, even from known synthetic M19 ciphertext.

## Programme implication

The continuous-emission model is too weakly identified to support a Voynich language claim. Lowering the recovery/reproducibility gates after seeing this result would be p-hacking. The next admissible direction must change the image representation or add independently observable graphical constraints; it cannot simply make the language-assisted fit more flexible.
