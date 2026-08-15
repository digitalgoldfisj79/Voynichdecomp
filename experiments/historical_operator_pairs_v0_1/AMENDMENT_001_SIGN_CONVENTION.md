# Amendment 001 — operator-vector sign convention

Date: 2026-08-15
Status: pre-inference, pre-Voynich

The committed protocol contains a sign-convention typo. It says the primary operator vector is `EXPANDED minus ABBREVIATED` while immediately stating that a negative `dH1` means abbreviation lowers H1. Those two statements are inconsistent: if abbreviation lowers H1, then `EXPANDED - ABBREVIATED` is positive.

No external metric inference and no Voynich scoring had been performed when this was noticed. Only archive/schema preflight had run.

For all v0.1 analyses the operator vector is therefore defined as:

`delta_metric = ABBREVIATED - EXPANDED`

Thus:
- negative delta_H1 = abbreviation lowers first-order conditional entropy;
- negative delta_H0 = abbreviation lowers unigram entropy;
- positive delta_(H0-H1) = abbreviation increases the H0-H1 gap.

All other preregistered gates and decision rules are unchanged. This amendment corrects notation only and does not use target information or observed external effect sizes.
