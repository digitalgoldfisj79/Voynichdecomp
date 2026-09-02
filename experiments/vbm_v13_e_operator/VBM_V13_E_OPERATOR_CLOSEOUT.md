# VBM v13 — e-operator geometry closeout

Date: 2026-09-02
Status: **COMPLETED**
Binding job: `6a97ccc021c5aa7c8364d848` (`cpu-upgrade`)
Execution-only failed precursor: `6a97cc7921c5aa7c8364d83c`
Protocol: `VBM_V13_E_OPERATOR_GEOMETRY_PROTOCOL.md`
Pre-binding addendum: `VBM_V13_PREBINDING_ADDENDUM.md`

## Frozen programme verdict

`V13_METHOD_NOT_QUALIFIED`

The discrete context-clustering/permutation pipeline failed its mandatory known-answer synthetic qualification. Therefore all Voynich Branch-A/B results are descriptive only and cannot support a global e-operator claim.

No plaintext, language likelihood, or GPU was used.

## Execution note

The first job failed before any scientific output because the inherited V12 dataclass module was loaded by dynamic `exec` without registration in `sys.modules`. This was a packaging error only. The scientific runner was left unchanged; an execution-only wrapper registered the remote module before `exec` and the binding job then completed normally.

## Synthetic method qualification

Binding calibration used six frozen V12 Stage-A POS replicates with a true shared global `pi`, paired with six matched `NUC_BROKEN` replicates.

Aggregate results:

- POS replicates with HOLD normalised accuracy >= 0.50: **2/6**; required >=5/6;
- median POS HOLD NACC: `0.4888411886`;
- median NUC_BROKEN HOLD NACC: `0.1269770880`;
- median separation: `0.3618641007`, exceeding the required 0.25;
- paired POS raw-accuracy wins over NUC_BROKEN: **5/6**, meeting that subcriterion.

The method therefore discriminated the generating architecture on average, but did not recover the known shared operator reliably enough across replicates. Because the qualification gate was conjunctive, the calibration failed.

POS HOLD NACC values were approximately:

- PEAKED-0: `0.495798`;
- PEAKED-1: `0.481884`;
- PEAKED-2: `0.547032`;
- MODERATE-0: `0.724138`;
- MODERATE-1: `0.215116`;
- MODERATE-2: `0.236742`.

This instability is the central negative result: context clustering does not reliably reconstruct a known repeated `e` operator even when one truly generated the synthetic corpus.

## Voynich output — descriptive only

Using the unchanged Q0b/V11 parser and H1/C1 firewalls:

- eligible nucleus types: **132**;
- TRAIN nucleus occurrences: **24,292**;
- INTERNAL_HOLDOUT nucleus occurrences: **6,687**;
- one-step e-ladder pairs: **18**;
- two-step e-ladder chains: **2**.

The TRAIN-selected model chose `K=2` with an identity permutation on the two contextual clusters.

Observed one-step performance:

- TRAIN raw accuracy: `1.0`;
- TRAIN NACC: `1.0`;
- HOLD raw accuracy: `1.0`;
- HOLD NACC: `1.0`.

However the full familywise matched-target null also reached a 99th percentile of `1.0`, and the empirical p-value was `0.02619738`, above the frozen `0.01` threshold. Thus the apparently perfect Voynich fit is non-selective under the preregistered null and Branch A fails even descriptively as an inferential gate:

`A_NO_SHARED_E_PERMUTATION`.

Branch B had only two eligible m->m+2 chains, below the required five:

`B_UNDERPOWERED_TWO_STEP`.

## Interpretation

V13 does not overturn V11-B. The replicated V11 result still supports unusually strong contextual similarity among e-ladder nuclei. What V13 fails to establish is the stronger claim that adding one `e` corresponds to a single repeated global discrete-state transformation shared across nucleus families.

Two separate reasons block that promotion:

1. the proposed clustering/permutation instrument cannot reliably recover a known true global operator in synthetic M12 data;
2. the Voynich perfect-looking K=2 identity result is itself non-selective against the matched familywise null.

Therefore the `e` family should presently be described as a reproducible compositional/graphotactic relation, not a recovered cipher operator.

## Permissible continuation

The next defensible question is mechanistic rather than another global-operator fit:

> Is the V11 e-ladder similarity explained by shared token-frame morphology `(left half, right half)`, or does it persist after conditioning on the same visible token frame?

That question follows directly from the two independent V11 positives: e-ladder contextual similarity (B) and factorised boundary halves (C). It can be tested ciphertext-only using the existing parser, without language or plaintext search.

No V14 is automatically authorised to decode Voynich. Any continuation must be a fresh preregistered frame-mediation programme.