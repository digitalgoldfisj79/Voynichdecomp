# VBM HMM v2 — Amendment 001: Moment-factorisation initializer

Date: 2026-08-11

The pre-Q0 engineering smoke of the random-Dirichlet HMM initializer failed on all three Bavarian smoke regimes. No formal Q0-HS run and no Voynich HMM target score had begun.

Observed failures showed the expected local-identifiability pathology of a 123-surface / 5-vowel emission model: independent random starts settled in incompatible soft emission basins even when the correct language transition matrix was supplied.

The scientific representation, 12-regime homophone sweep, target split, language panel and gates remain unchanged. The initializer is replaced prospectively before Q0-HS by a global moment fit:

1. Compute the empirical within-sequence observed bigram joint matrix `P_obs(y_t,y_{t+1})` on the fit ciphertext.
2. For candidate language L, form hidden-letter bigram joint `J_L = diag(pi_L) T_L`, normalized to sum 1.
3. Fit typed row-stochastic emission matrix E to minimise Frobenius loss
   `|| E^T J_L E - P_obs ||_F^2`
   under exact core->consonant and bridge->vowel zero constraints.
4. Optimise typed emission logits with deterministic Adam from four fresh starts (1,500 moment steps; learning rate 0.05).
5. Use each resulting E as the corresponding initialization for the already-frozen forward-backward EM refinement.
6. A/B ensemble construction and every recovery, convergence, language, negative-control and target gate remain unchanged.

This is an inference-engine calibration amendment triggered solely by synthetic smoke failure. If the moment-initialized smoke still fails, v2 will not proceed to formal Q0-HS.
