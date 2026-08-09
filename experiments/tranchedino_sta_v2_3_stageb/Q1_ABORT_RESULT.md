# Tranchedino × STA v2.3 — Q1 abort and Stage-B closure

Date: 2026-08-09
Implementation freeze: `47a75db0773d029c9782b1e03523c230322ca9fe`
Binding solver commit: `fd225442d3e84950bc997f405b9d6552e4965873`
Q1 control: replicate 0, `p_null=0.01`

## Formal result

**B1 INSTRUMENT NOT QUALIFIED.**

The first untouched Q1 control irrecoverably violates several preregistered minimum gates. Because the gates include per-control minima, no result from the remaining eleven controls can repair the verdict. They were therefore not run.

No Voynich mixed-unit target fit, score, plaintext or T23/H23/C23 split was generated.

## Binding Q1 result

| metric | Q1-0 result | frozen requirement | status |
|---|---:|---:|---|
| plaintext recovery | **0.90525** | minimum >=0.75; median later >=0.90 | pass on this control |
| occurrence-weighted semantic recovery | **0.93028** | minimum >=0.70 | pass |
| alphabet occurrence recovery | **0.95662** | diagnostic | strong |
| geminate occurrence recovery | **0.00000** | minimum >=0.70 | **FAIL** |
| nomenclator occurrence-word recovery | **0.08333** | minimum >=0.60 | **FAIL** |
| null occurrence F1 | **0.21465** | minimum >=0.75 | **FAIL** |
| held-out surface coverage | **0.99908** | >=0.95 | pass |
| A/B semantic-map agreement | **0.98521** | >=0.85 | pass |
| A/B fit-score difference | **0.003505 nats/char** | <=0.0001 | **FAIL** |

Ensemble A selected search score `-2.37855596` nats/cipher-event; ensemble B `-2.38092182`. Their maps agree on 98.52% of occurrence weight, but the character-score difference is thirty-five times the convergence tolerance.

Frozen-map held-out character score for the selected A map: `-2.52878163` nats/decoded character, at 99.91% surface coverage. This score does **not** become a historical-control floor because the instrument failed qualification first.

## Interpretation

The result cleanly separates two facts that would otherwise be easy to conflate:

1. **Most of the plaintext is recoverable.** Overall character recovery is 90.5% and overall occurrence-weighted semantic recovery is 93.0%.
2. **The specifically mixed historical components are not recoverable by the frozen instrument.** Alphabetic signs carry the success (95.7% occurrence recovery), while the f.69v additions that distinguish Stage B from Stage A fail badly: 0% geminate recovery, 8.3% nomenclator occurrence-word recovery and null F1 0.215.

Thus a high aggregate plaintext score would be misleading evidence for the mixed Tranchedino architecture. The solver can recover the dominant alphabetic layer while remaining effectively unable to identify the additional historical code classes.

This reproduces, at a much larger data scale and under a more historically exact key geometry, the central identifiability warning from the old nomenclator work: high expanded-character recovery does not imply recovery of sparse whole-word code semantics.

## Scientific consequence

The exact **f.69v one-sign mixed-unit model** is closed as a Voynich test under v2.3:

`B1 INSTRUMENT NOT QUALIFIED / NO VOYNICH INFERENCE ADMISSIBLE`.

The following remain untouched and are not ruled out by this result:

- a separately audited Tranchedino key with a genuine syllabary/substring inventory;
- a variable-sign segmentation model tied to connected `aaa` rather than one full-STA member per historical sign;
- visually distinct historical sign structure not represented as one RF full-STA member;
- historical codebook semantics outside the frozen top-96 Paduan word model.

Those are separate hypotheses and may not be obtained by repairing v2.3 after Q1. Any continuation must start with a new prospective historical template/instrument, not alter the failed thresholds or Q1 solver.

## Compute stop

The remaining eleven Q1 controls, specificity controls, label-invariance audit and all Voynich target stages remain unrun. No Hugging Face jobs are running at closure.
