# Amadi Residuals v1 — Q0 Source Reconciliation Addendum

Date: 2026-08-11
Status: **FROZEN BEFORE Q1 / ANY VOYNICH H2 SCORE**

This addendum records source-internal discrepancies discovered by the first smoke run. No target likelihood had been computed.

## VC_END / section 0013

The prose rule is unambiguous: remove vowels from the middle of each word and append **all vowels in their original order**. Applying that rule mechanically to the 34 word pairs transcribed in the worked example reproduces 28 pairs exactly.

Six tabulated pairs disagree with the stated rule:

| plaintext | mechanical prose-rule output | transcribed worked output | discrepancy |
|---|---|---|---|
| discorere | `dscrrioee` | `dscrioee` | one `r` omitted |
| differentia | `dffrntieeia` | `dffrntieia` | one `e` omitted |
| riputatione | `rpttniuaioe` | `rpttnuaioe` | initial moved `i` omitted |
| melanconia | `mlncneaoia` | `mlnceaoia` | one `n` omitted |
| litterati | `lttrtieai` | `lttrteai` | one `i` omitted |
| dellandar | `dllndreaa` | `dlIndreaa` | one glyph is transcribed as capital `I` where the mechanical output has `l` |

These deviations do **not** define a coherent alternative algorithm; five are single-character omissions and one is a transcriptional glyph ambiguity. The computational arm therefore implements the explicit prose operation and retains these six pairs as `SOURCE_EXAMPLE_DISCREPANCY`, rather than tuning the transform to reproduce errors.

This is not silent correction: qualification reports the 28/34 exact example-conformity count and the six fixed discrepancies.

## R12_V1_024

The short examples in section 024 demonstrate individual orthographic replacement rules one at a time. They are not outputs of simultaneously applying every listed reduction. For example, `grande -> grante` illustrates `d -> t`; applying the full composite formalisation would also act on `g`. Likewise `mouendo -> mooendo` illustrates consonantal-u replacement without simultaneously applying the independent `d -> t` rule.

Q0 therefore tests each quoted short pair against **its source-local rule**, not against the full composite formalisation.

The programme's `R12_V1_024` computational transform is a deterministic composition of the explicitly listed rules. Because Amadi describes orthographic alteration as a flexible security principle and the long worked reduced text is not mechanically uniform, this computational rule is henceforth labelled:

`SOURCE-DERIVED DETERMINISTIC FORMALISATION`, not a claim that every character of Amadi's long worked example follows a single automatic rewrite machine.

The broad 19→12 `R12H` surface remains, as already stated, a conservative stress-test superfamily rather than an exact historical table.

## Q0 decision rule frozen here

`VC_END` passes source fidelity if:

1. the prose algorithm is implemented literally;
2. exactly the six discrepancies above are reproduced as discrepancies (no more, no fewer);
3. all other 28 tabulated word pairs match.

`R12_V1_024` passes source fidelity if every individually quoted replacement example is reproduced under the specific rule it illustrates, and the composed computational formalisation maps arbitrary supported Italian words only into the declared 12-value alphabet.

Any additional source mismatch discovered after this freeze blocks the affected arm rather than changing this rule.