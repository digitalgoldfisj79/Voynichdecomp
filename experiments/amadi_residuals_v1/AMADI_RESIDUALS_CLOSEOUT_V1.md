# Amadi Residuals v1 — Programme Closeout

Date: 2026-08-11
Branch: `experiment/amadi-residuals-v1-20260811`
Status: **STOPPED AT PREREGISTERED H2 DECISION POINT**

## Executive result

The programme found **no admissible Voynich candidate** among the bounded Amadi-derived residual mechanisms.

One family, `VC_END`, is legitimately closed negative on held-out Voynich because its target solver converged and the H2 score failed both the frozen absolute positive-control floor and the matched-M0 delta floor by large margins.

Three families—`R12H`, `PWA_K`, and `GHOUSE5`—also scored below their relevant positive floors and failed their positive-specific gates, but their full-target optimizers did not satisfy the preregistered A/B convergence criterion. They are therefore **UNRESOLVED_SEARCH**, not formal negative family results. The protocol prohibits increasing optimizer budget or changing the solver after seeing H2.

No family reaches C2. C2 remains completely sealed.

## What was learned before target testing

### Historical scope

The exact Amadi residuals tested here remain later-Renaissance mechanisms under the current provenance audit. No exact admitted mechanism was upgraded to a secure <=1450 H0/H1 operational witness. Some components—letter-count alteration, homophony, numerical/coordinate devices—have older antecedents, but not the exact residual constructions.

Therefore this programme was a deliberate anachronism/wider-repertoire stress-test. Its outcome does not weaken the previous circa-1400–1450 cipher closeout.

### Source audit

The run corrected two overstatements in the old Amadi notes before qualification:

1. Section 397 supports one-letter syllable mutation, not specifically mutation triggered by token reuse.
2. `g` in section 024 is deleted in the worked examples; it is not transformed into an additional `i`.

The section-013 worked table also contains six fixed deviations from its explicit prose algorithm; these were preserved as source discrepancies rather than used to alter the computational rule.

### Structural eliminations

Two exact Amadi systems were closed before statistics:

- literal NTRC/DBAC output uses <=4 surface characters and cannot achieve the frozen 0.995 RF coverage without an unlicensed glyph collapse;
- literal modulo-105 output uses only seven digits and likewise cannot achieve 0.995 RF coverage.

Autokey and walking/two-stream arms were not admitted because the supplied source extraction did not determine one unique direct-RF executable inverse. Dual-meaning systems remained non-identifiable without an externally fixed second text/key.

## Qualification result

All four admitted primary families passed recovery qualification.

Q2 blind recognition:

- family accuracy: 1.000
- language accuracy: 1.000
- PWA exact-rule accuracy: 1.000
- median recovery: 1.000

Q4 specificity:

- 0/80 structured negatives false-positive
- iid 0/16
- order-2 Markov 0/16
- motif-repeat/mutate 0/16
- copy/mutate 0/16
- slot grammar 0/16

Thus the H2 outcome is not attributable to an instrument that had already failed positive controls.

## H2 target result

| family | result |
|---|---|
| `VC_END` | **CLOSED NEGATIVE / INCOMPATIBLE UNDER v1** |
| `R12H` | **UNRESOLVED_SEARCH; NO CANDIDATE** |
| `PWA_K` | **UNRESOLVED_SEARCH; NO CANDIDATE** |
| `GHOUSE5` | **UNRESOLVED_SEARCH; NO CANDIDATE** |

All three unresolved families were below their absolute and matched-baseline positive floors. This is descriptively adverse to the hypotheses, but it cannot be converted into formal rejection because the target fits did not converge.

PWA additionally failed its word-reset specificity floor. GHOUSE5 had adequate data in all selector states and the real selector labels beat within-folio label permutations, but its five fitted maps were radically unstable across independent ensembles and the family remained far below the calibrated positive floors. The permutation result therefore cannot be interpreted as cipher evidence.

## Binding scientific conclusion

**AMADI RESIDUALS v1: NO POSITIVE VOYNICH CIPHER EVIDENCE.**

More precisely:

1. The exact vowel-to-word-end Amadi mechanism is rejected under the qualified v1 test.
2. No reduced-alphabet/homophonic, positional multi-alphabet, or gallows-house model passes the H2 gates.
3. Those three broader families remain computationally unresolved under the frozen target optimizer, rather than empirically supported.
4. C2 remains sealed because no admissible H2 candidate exists.
5. The prior c.1400–1450 historical cipher closeout remains intact independently of this result, because none of these exact later mechanisms was upgraded to H0/H1.

## What v1 does NOT license

Do not claim:

- that all Amadi-style ciphers have been disproved;
- that R12H/PWA/GHOUSE are formally rejected;
- that gallows are house selectors because their real labels beat permutations;
- that the nonconverged best mappings are plaintext;
- that a larger annealing budget may now be tried as a rescue;
- that C2 may be inspected;
- that the late-sixteenth-century Amadi repertoire proves a fifteenth-century transmission path.

## Reopening conditions

A new Amadi-derived cipher programme is justified only by one of:

1. a materially new inference algorithm, frozen and qualified on fresh controls before any reuse of H2/C2;
2. a concrete <=1450 operational witness that independently fixes a currently later-only mechanism or state schedule;
3. a newly recovered exact Amadi worked example/key that makes an underdetermined autokey/walking arm executable;
4. independent manuscript evidence fixing a selector, house schedule, reduced alphabet, or latent glyph grouping before target fitting.

Merely increasing restarts/annealing proposals or trying alternative target transliterations after this H2 result is not a valid v1 continuation.

## Compute closeout

Qualification job: `6a7b55d627caad61c6eac050` — completed.
Target H2 job: `6a7b583627caad61c6eac084` — completed.
Post-run Hugging Face process check: no running jobs.

No paid job was left orphaned.
