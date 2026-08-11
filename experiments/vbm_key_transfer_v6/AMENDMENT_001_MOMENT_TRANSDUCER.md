# Amendment 001 — moment transducer as the v6 key instrument

Date: 2026-08-11
Status: FROZEN BEFORE Q0 CONTROL SCORING

The protocol phrase “fit the existing moment-initialised typed soft-HMM emission matrix” is narrowed for v6 to the **moment-factorisation emission matrix before Baum–Welch**.

Reason: v6 is testing cross-folio reuse of a surface→latent transducer, not absolute sequence-language likelihood. v5 showed target Baum–Welch convergence can fail even when the moment geometry is stable. Continuing EM would therefore add an optimiser-specific failure mode unrelated to the v6 causal question.

For a candidate language with stationary bigram joint matrix `J = diag(pi) T`, fit typed emission matrix `E` on train surface bigram frequencies `P_train` by the frozen moment objective:

`min_E || E^T J E - P_train ||_F^2`,

with core letters restricted to the 21 core observations and vowels restricted to the 123 bridge observations.

Language selection uses train moment loss only. The selected frozen transducer predicts held-out surface bigram probabilities `M = E^T J E`. Held-out score is mean bigram log probability under `M` with a fixed `1e-12` numerical floor and renormalisation. The primary ITE remains observed held-out score minus the median score of topology-preserving label permutations under the identical frozen `M`.

Map stability is computed from two independently fitted moment matrices on disjoint train-folio halves. Posterior surface→latent labels use `argmax_s pi[s] E[s,y]`.

All CAL/VAL thresholds and stop rules in the main protocol remain unchanged. This amendment reduces compute and removes post-moment EM; it does not alter the target representation, permutation null, control families, C1 seal, or decision gates.
